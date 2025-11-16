// Function: sub_D76140
// Address: 0xd76140
//
char *__fastcall sub_D76140(char *a1, char *a2, _QWORD *a3)
{
  char *result; // rax
  __int64 v5; // rdx
  __int64 v6; // rsi
  __int64 v7; // rdx
  char *v8; // rsi
  unsigned __int64 v9; // rdx
  unsigned __int64 v10; // rdx

  result = a2;
  v5 = a2 - a1;
  v6 = (a2 - a1) >> 5;
  v7 = v5 >> 3;
  if ( v6 > 0 )
  {
    v8 = &a1[32 * v6];
    v9 = *a3 & 0xFFFFFFFFFFFFFFF8LL;
    while ( v9 != (*(_QWORD *)a1 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      if ( v9 == (*((_QWORD *)a1 + 1) & 0xFFFFFFFFFFFFFFF8LL) )
        return a1 + 8;
      if ( v9 == (*((_QWORD *)a1 + 2) & 0xFFFFFFFFFFFFFFF8LL) )
        return a1 + 16;
      if ( v9 == (*((_QWORD *)a1 + 3) & 0xFFFFFFFFFFFFFFF8LL) )
        return a1 + 24;
      a1 += 32;
      if ( v8 == a1 )
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
    v10 = *a3 & 0xFFFFFFFFFFFFFFF8LL;
    goto LABEL_21;
  }
  if ( v7 == 3 )
  {
    v10 = *a3 & 0xFFFFFFFFFFFFFFF8LL;
    if ( v10 == (*(_QWORD *)a1 & 0xFFFFFFFFFFFFFFF8LL) )
      return a1;
    a1 += 8;
LABEL_21:
    if ( v10 != (*(_QWORD *)a1 & 0xFFFFFFFFFFFFFFF8LL) )
    {
      a1 += 8;
      goto LABEL_17;
    }
    return a1;
  }
  if ( v7 != 1 )
    return result;
  v10 = *a3 & 0xFFFFFFFFFFFFFFF8LL;
LABEL_17:
  if ( v10 == (*(_QWORD *)a1 & 0xFFFFFFFFFFFFFFF8LL) )
    return a1;
  return result;
}
