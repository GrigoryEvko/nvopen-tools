// Function: sub_1268290
// Address: 0x1268290
//
_BYTE *__fastcall sub_1268290(_QWORD *a1, _BYTE *a2, size_t a3)
{
  size_t v5; // rax
  _QWORD *v6; // rdx
  _BYTE *v7; // rdi
  _BYTE *result; // rax
  size_t v9; // rsi
  __int64 v10; // rcx
  __int64 v11; // rdi
  size_t v12; // rdx
  _QWORD *v13; // rdi
  size_t v14; // [rsp+8h] [rbp-48h] BYREF
  _QWORD *v15; // [rsp+10h] [rbp-40h] BYREF
  size_t n; // [rsp+18h] [rbp-38h]
  _QWORD src[6]; // [rsp+20h] [rbp-30h] BYREF

  if ( !a2 )
  {
    LOBYTE(src[0]) = 0;
    v7 = (_BYTE *)a1[30];
    v12 = 0;
    v15 = src;
LABEL_10:
    a1[31] = v12;
    v7[v12] = 0;
    result = v15;
    goto LABEL_11;
  }
  v14 = a3;
  v15 = src;
  v5 = a3;
  if ( a3 > 0xF )
  {
    v15 = (_QWORD *)sub_22409D0(&v15, &v14, 0);
    v13 = v15;
    src[0] = v14;
LABEL_15:
    memcpy(v13, a2, a3);
    v5 = v14;
    v6 = v15;
    goto LABEL_5;
  }
  if ( a3 == 1 )
  {
    LOBYTE(src[0]) = *a2;
    v6 = src;
    goto LABEL_5;
  }
  if ( a3 )
  {
    v13 = src;
    goto LABEL_15;
  }
  v6 = src;
LABEL_5:
  n = v5;
  *((_BYTE *)v6 + v5) = 0;
  v7 = (_BYTE *)a1[30];
  result = v7;
  if ( v15 == src )
  {
    v12 = n;
    if ( n )
    {
      if ( n == 1 )
        *v7 = src[0];
      else
        memcpy(v7, src, n);
      v12 = n;
      v7 = (_BYTE *)a1[30];
    }
    goto LABEL_10;
  }
  v9 = n;
  v10 = src[0];
  if ( v7 == (_BYTE *)(a1 + 32) )
  {
    a1[30] = v15;
    a1[31] = v9;
    a1[32] = v10;
  }
  else
  {
    v11 = a1[32];
    a1[30] = v15;
    a1[31] = v9;
    a1[32] = v10;
    if ( result )
    {
      v15 = result;
      src[0] = v11;
      goto LABEL_11;
    }
  }
  v15 = src;
  result = src;
LABEL_11:
  n = 0;
  *result = 0;
  if ( v15 != src )
    return (_BYTE *)j_j___libc_free_0(v15, src[0] + 1LL);
  return result;
}
