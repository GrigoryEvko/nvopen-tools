// Function: sub_19C03B0
// Address: 0x19c03b0
//
__int64 __fastcall sub_19C03B0(__int64 a1, char **a2)
{
  char *v2; // r12
  char *v3; // rcx
  __int64 v4; // rdx
  __int64 result; // rax
  __int64 v6; // rdx
  char *v7; // rdx
  char *v8; // rax
  char *v9; // rcx

  v2 = a2[1];
  v3 = *a2;
  v4 = v2 - *a2;
  result = v4 >> 5;
  v6 = v4 >> 3;
  if ( result > 0 )
  {
    result = (__int64)&v3[32 * result];
    while ( *(_QWORD *)v3 != a1 )
    {
      if ( *((_QWORD *)v3 + 1) == a1 )
      {
        v3 += 8;
        goto LABEL_8;
      }
      if ( *((_QWORD *)v3 + 2) == a1 )
      {
        v3 += 16;
        goto LABEL_8;
      }
      if ( *((_QWORD *)v3 + 3) == a1 )
      {
        v3 += 24;
        goto LABEL_8;
      }
      v3 += 32;
      if ( (char *)result == v3 )
      {
        v6 = (v2 - v3) >> 3;
        goto LABEL_20;
      }
    }
    goto LABEL_8;
  }
LABEL_20:
  if ( v6 == 2 )
  {
LABEL_27:
    if ( *(_QWORD *)v3 != a1 )
    {
      v3 += 8;
      goto LABEL_23;
    }
    goto LABEL_8;
  }
  if ( v6 != 3 )
  {
    if ( v6 != 1 )
      return result;
LABEL_23:
    if ( *(_QWORD *)v3 != a1 )
      return result;
    goto LABEL_8;
  }
  if ( *(_QWORD *)v3 != a1 )
  {
    v3 += 8;
    goto LABEL_27;
  }
LABEL_8:
  if ( v2 != v3 )
  {
    result = (__int64)(v3 + 8);
    if ( v2 == v3 + 8 )
      goto LABEL_14;
    do
    {
      if ( *(_QWORD *)result != a1 )
      {
        *(_QWORD *)v3 = *(_QWORD *)result;
        v3 += 8;
      }
      result += 8;
    }
    while ( v2 != (char *)result );
    if ( v2 != v3 )
    {
LABEL_14:
      v7 = a2[1];
      if ( v2 != v7 )
      {
        v8 = (char *)memmove(v3, v2, v7 - v2);
        v7 = a2[1];
        v3 = v8;
      }
      result = v7 - v2;
      v9 = &v3[v7 - v2];
      if ( v9 != v7 )
        a2[1] = v9;
    }
  }
  return result;
}
