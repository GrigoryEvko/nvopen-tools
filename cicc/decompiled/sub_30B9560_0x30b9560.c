// Function: sub_30B9560
// Address: 0x30b9560
//
char *__fastcall sub_30B9560(char *a1, char *a2, __int64 a3)
{
  char *result; // rax
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // rdx
  __int64 v8; // rcx
  char *v9; // rsi
  __int64 v10; // r8

  result = a2;
  v5 = a2 - a1;
  v6 = (a2 - a1) >> 5;
  v7 = v5 >> 3;
  if ( v6 > 0 )
  {
    v8 = a3 + 8;
    v9 = &a1[32 * v6];
    while ( v8 != **(_QWORD **)a1 + 8LL )
    {
      if ( v8 == **((_QWORD **)a1 + 1) + 8LL )
        return a1 + 8;
      if ( v8 == **((_QWORD **)a1 + 2) + 8LL )
        return a1 + 16;
      if ( v8 == **((_QWORD **)a1 + 3) + 8LL )
        return a1 + 24;
      a1 += 32;
      if ( a1 == v9 )
      {
        v7 = (result - a1) >> 3;
        goto LABEL_10;
      }
    }
    return a1;
  }
LABEL_10:
  if ( v7 == 2 )
  {
    v10 = a3 + 8;
    goto LABEL_21;
  }
  if ( v7 == 3 )
  {
    v10 = a3 + 8;
    if ( v10 == **(_QWORD **)a1 + 8LL )
      return a1;
    a1 += 8;
LABEL_21:
    if ( v10 != **(_QWORD **)a1 + 8LL )
    {
      a1 += 8;
      goto LABEL_17;
    }
    return a1;
  }
  if ( v7 != 1 )
    return result;
  v10 = a3 + 8;
LABEL_17:
  if ( v10 == **(_QWORD **)a1 + 8LL )
    return a1;
  return result;
}
