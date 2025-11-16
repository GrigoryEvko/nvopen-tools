// Function: sub_CCA250
// Address: 0xcca250
//
__int64 __fastcall sub_CCA250(_QWORD *a1, void **a2)
{
  _BYTE *v3; // rdi
  __int64 v4; // rdx
  size_t v5; // rcx
  __int64 v6; // rsi
  _QWORD *v7; // rdi
  __int64 result; // rax
  size_t v9; // rdx
  _QWORD *v10; // [rsp+0h] [rbp-50h] BYREF
  size_t n; // [rsp+8h] [rbp-48h]
  _QWORD src[4]; // [rsp+10h] [rbp-40h] BYREF
  __int64 v13; // [rsp+30h] [rbp-20h]

  sub_CC9F70((__int64)&v10, a2);
  v3 = (_BYTE *)*a1;
  if ( v10 == src )
  {
    v9 = n;
    if ( n )
    {
      if ( n == 1 )
        *v3 = src[0];
      else
        memcpy(v3, src, n);
      v9 = n;
      v3 = (_BYTE *)*a1;
    }
    a1[1] = v9;
    v3[v9] = 0;
    v3 = v10;
  }
  else
  {
    v4 = src[0];
    v5 = n;
    if ( v3 == (_BYTE *)(a1 + 2) )
    {
      *a1 = v10;
      a1[1] = v5;
      a1[2] = v4;
    }
    else
    {
      v6 = a1[2];
      *a1 = v10;
      a1[1] = v5;
      a1[2] = v4;
      if ( v3 )
      {
        v10 = v3;
        src[0] = v6;
        goto LABEL_5;
      }
    }
    v10 = src;
    v3 = src;
  }
LABEL_5:
  n = 0;
  *v3 = 0;
  v7 = v10;
  a1[4] = src[2];
  a1[5] = src[3];
  result = v13;
  a1[6] = v13;
  if ( v7 != src )
    return j_j___libc_free_0(v7, src[0] + 1LL);
  return result;
}
