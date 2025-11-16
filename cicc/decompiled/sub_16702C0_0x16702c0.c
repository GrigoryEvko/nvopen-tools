// Function: sub_16702C0
// Address: 0x16702c0
//
_QWORD *__fastcall sub_16702C0(_QWORD *a1, _BYTE *a2, __int64 a3)
{
  _BYTE *v3; // r12
  _BYTE *v5; // rdi
  _QWORD *result; // rax
  __int64 v7; // rcx
  size_t v8; // rsi
  __int64 v9; // rdi
  __int64 v10; // r13
  unsigned __int64 v11; // r14
  unsigned __int64 v12; // rdx
  size_t v13; // rdx
  _QWORD *v14; // [rsp+0h] [rbp-40h] BYREF
  size_t n; // [rsp+8h] [rbp-38h]
  _QWORD src[6]; // [rsp+10h] [rbp-30h] BYREF

  v3 = a1 + 13;
  if ( !a2 )
  {
    LOBYTE(src[0]) = 0;
    v5 = (_BYTE *)a1[11];
    v13 = 0;
    v14 = src;
LABEL_19:
    a1[12] = v13;
    v5[v13] = 0;
    result = v14;
    goto LABEL_6;
  }
  v14 = src;
  sub_1670060((__int64 *)&v14, a2, (__int64)&a2[a3]);
  v5 = (_BYTE *)a1[11];
  result = (_QWORD *)a1[11];
  if ( v14 == src )
  {
    v13 = n;
    if ( n )
    {
      if ( n == 1 )
        *v5 = src[0];
      else
        memcpy(v5, src, n);
      v13 = n;
      v5 = (_BYTE *)a1[11];
    }
    goto LABEL_19;
  }
  v7 = src[0];
  v8 = n;
  if ( v5 == v3 )
  {
    a1[11] = v14;
    a1[12] = v8;
    a1[13] = v7;
  }
  else
  {
    v9 = a1[13];
    a1[11] = v14;
    a1[12] = v8;
    a1[13] = v7;
    if ( result )
    {
      v14 = result;
      src[0] = v9;
      goto LABEL_6;
    }
  }
  v14 = src;
  result = src;
LABEL_6:
  n = 0;
  *(_BYTE *)result = 0;
  if ( v14 != src )
    result = (_QWORD *)j_j___libc_free_0(v14, src[0] + 1LL);
  v10 = a1[12];
  if ( v10 )
  {
    result = (_QWORD *)a1[11];
    if ( *((_BYTE *)result + v10 - 1) != 10 )
    {
      v11 = v10 + 1;
      if ( result == (_QWORD *)v3 )
        v12 = 15;
      else
        v12 = a1[13];
      if ( v11 > v12 )
      {
        sub_2240BB0(a1 + 11, a1[12], 0, 0, 1);
        result = (_QWORD *)a1[11];
      }
      *((_BYTE *)result + v10) = 10;
      result = (_QWORD *)a1[11];
      a1[12] = v11;
      *((_BYTE *)result + v10 + 1) = 0;
    }
  }
  return result;
}
