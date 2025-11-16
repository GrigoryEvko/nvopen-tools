// Function: sub_13E5700
// Address: 0x13e5700
//
char *__fastcall sub_13E5700(char *a1, char *a2, __int64 *a3)
{
  char *result; // rax
  __int64 v5; // rcx
  __int64 v6; // rdx
  __int64 v7; // rdx
  char *v8; // rcx
  char *i; // rdx
  __int64 v10; // rdx

  result = a1;
  v5 = (a2 - a1) >> 5;
  v6 = (a2 - a1) >> 3;
  if ( v5 > 0 )
  {
    v7 = *a3;
    v8 = &a1[32 * v5];
    while ( *(_QWORD *)result != v7 )
    {
      if ( v7 == *((_QWORD *)result + 1) )
      {
        result += 8;
        goto LABEL_8;
      }
      if ( v7 == *((_QWORD *)result + 2) )
      {
        result += 16;
        goto LABEL_8;
      }
      if ( v7 == *((_QWORD *)result + 3) )
      {
        result += 24;
        goto LABEL_8;
      }
      result += 32;
      if ( v8 == result )
      {
        v6 = (a2 - result) >> 3;
        goto LABEL_15;
      }
    }
    goto LABEL_8;
  }
LABEL_15:
  if ( v6 == 2 )
  {
    v10 = *a3;
LABEL_25:
    if ( v10 == *(_QWORD *)result )
      goto LABEL_8;
    result += 8;
    goto LABEL_22;
  }
  if ( v6 == 3 )
  {
    v10 = *a3;
    if ( *(_QWORD *)result == *a3 )
      goto LABEL_8;
    result += 8;
    goto LABEL_25;
  }
  if ( v6 != 1 )
    return a2;
  v10 = *a3;
LABEL_22:
  if ( v10 != *(_QWORD *)result )
    return a2;
LABEL_8:
  if ( a2 != result )
  {
    for ( i = result + 8; a2 != i; i += 8 )
    {
      if ( *(_QWORD *)i != *a3 )
      {
        *(_QWORD *)result = *(_QWORD *)i;
        result += 8;
      }
    }
  }
  return result;
}
