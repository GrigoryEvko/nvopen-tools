// Function: sub_3223650
// Address: 0x3223650
//
__int64 __fastcall sub_3223650(char **a1, int a2)
{
  char *v2; // r12
  char *v3; // rcx
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 result; // rax
  char *v7; // rdx
  char *v8; // rdx
  char *v9; // rdx
  char *v10; // rax
  char *v11; // rcx

  v2 = a1[1];
  v3 = *a1;
  v4 = v2 - *a1;
  v5 = v4 >> 5;
  result = v4 >> 3;
  if ( v5 > 0 )
  {
    result = a2;
    v7 = &v3[32 * v5];
    while ( a2 != *(_QWORD *)v3 )
    {
      if ( a2 == *((_QWORD *)v3 + 1) )
      {
        v3 += 8;
        goto LABEL_8;
      }
      if ( a2 == *((_QWORD *)v3 + 2) )
      {
        v3 += 16;
        goto LABEL_8;
      }
      if ( a2 == *((_QWORD *)v3 + 3) )
      {
        v3 += 24;
        goto LABEL_8;
      }
      v3 += 32;
      if ( v7 == v3 )
      {
        result = (v2 - v3) >> 3;
        goto LABEL_20;
      }
    }
    goto LABEL_8;
  }
LABEL_20:
  if ( result == 2 )
  {
    result = a2;
LABEL_29:
    if ( *(_QWORD *)v3 != result )
    {
      v3 += 8;
      goto LABEL_24;
    }
    goto LABEL_8;
  }
  if ( result != 3 )
  {
    if ( result != 1 )
      return result;
    result = a2;
LABEL_24:
    if ( *(_QWORD *)v3 != result )
      return result;
    goto LABEL_8;
  }
  result = a2;
  if ( *(_QWORD *)v3 != a2 )
  {
    v3 += 8;
    goto LABEL_29;
  }
LABEL_8:
  if ( v2 != v3 )
  {
    v8 = v3 + 8;
    if ( v2 == v3 + 8 )
      goto LABEL_14;
    do
    {
      if ( *(_QWORD *)v8 != result )
      {
        *(_QWORD *)v3 = *(_QWORD *)v8;
        v3 += 8;
      }
      v8 += 8;
    }
    while ( v8 != v2 );
    if ( v2 != v3 )
    {
LABEL_14:
      v9 = a1[1];
      if ( v2 != v9 )
      {
        v10 = (char *)memmove(v3, v2, v9 - v2);
        v9 = a1[1];
        v3 = v10;
      }
      result = v9 - v2;
      v11 = &v3[v9 - v2];
      if ( v11 != v9 )
        a1[1] = v11;
    }
  }
  return result;
}
