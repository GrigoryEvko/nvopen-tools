// Function: sub_127BDC0
// Address: 0x127bdc0
//
__int64 *__fastcall sub_127BDC0(__int64 *a1, _QWORD *a2, __int64 a3)
{
  __int64 *v5; // rbx
  __int64 v6; // r15
  size_t v7; // rax
  __int64 v9; // rdx
  __int64 *v10; // rdi
  __int64 v11; // rdx
  size_t v12; // rcx
  __int64 v13; // rsi
  size_t v14; // rdx
  __int64 *v15; // [rsp+0h] [rbp-50h] BYREF
  size_t n; // [rsp+8h] [rbp-48h]
  _QWORD src[8]; // [rsp+10h] [rbp-40h] BYREF

  v5 = a1 + 2;
  *a1 = (__int64)(a1 + 2);
  sub_127A7E0(a1, (_BYTE *)*a2, *a2 + a2[1]);
  if ( (*(_BYTE *)(a3 + 202) & 0x40) != 0 )
    return a1;
  if ( (*(_BYTE *)(a3 + 199) & 1) != 0 )
  {
    v6 = *(_QWORD *)(a3 + 136);
    if ( v6 )
    {
      v7 = strlen(*(const char **)(a3 + 136));
      sub_2241130(a1, 0, a1[1], v6, v7);
      return a1;
    }
  }
  if ( (*(_BYTE *)(a3 + 197) & 0x60) == 0 || (v9 = *(_QWORD *)(a3 + 128)) == 0 || (*(_BYTE *)(v9 + 198) & 0x20) == 0 )
  {
    if ( (*(_BYTE *)(a3 + 198) & 0x20) == 0 )
      return a1;
    v9 = a3;
  }
  sub_127AC30((__int64)&v15, a2, v9, 0);
  v10 = (__int64 *)*a1;
  if ( v15 == src )
  {
    v14 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)v10 = src[0];
      else
        memcpy(v10, src, n);
      v14 = n;
      v10 = (__int64 *)*a1;
    }
    a1[1] = v14;
    *((_BYTE *)v10 + v14) = 0;
    v10 = v15;
    goto LABEL_15;
  }
  v11 = src[0];
  v12 = n;
  if ( v5 == v10 )
  {
    *a1 = (__int64)v15;
    a1[1] = v12;
    a1[2] = v11;
  }
  else
  {
    v13 = a1[2];
    *a1 = (__int64)v15;
    a1[1] = v12;
    a1[2] = v11;
    if ( v10 )
    {
      v15 = v10;
      src[0] = v13;
      goto LABEL_15;
    }
  }
  v15 = src;
  v10 = src;
LABEL_15:
  n = 0;
  *(_BYTE *)v10 = 0;
  if ( v15 != src )
    j_j___libc_free_0(v15, src[0] + 1LL);
  return a1;
}
