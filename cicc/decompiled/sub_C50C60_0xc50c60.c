// Function: sub_C50C60
// Address: 0xc50c60
//
__int64 __fastcall sub_C50C60(__int64 a1, __int16 a2, __int64 a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  _QWORD *v7; // rdi
  _QWORD *v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rdx
  bool v11; // zf
  size_t v13; // rdx
  void *dest; // [rsp+0h] [rbp-60h] BYREF
  size_t v15; // [rsp+8h] [rbp-58h]
  _QWORD v16[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD *v17; // [rsp+20h] [rbp-40h] BYREF
  size_t n; // [rsp+28h] [rbp-38h]
  _QWORD src[6]; // [rsp+30h] [rbp-30h] BYREF

  dest = v16;
  v15 = 0;
  LOBYTE(v16[0]) = 0;
  if ( !a5 )
  {
    LOBYTE(src[0]) = 0;
    v13 = 0;
    v7 = v16;
    v17 = src;
LABEL_15:
    v15 = v13;
    *((_BYTE *)v7 + v13) = 0;
    v8 = v17;
    goto LABEL_6;
  }
  v17 = src;
  sub_C4FB50((__int64 *)&v17, a5, (__int64)&a5[a6]);
  v7 = dest;
  v8 = dest;
  if ( v17 == src )
  {
    v13 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v13 = n;
      v7 = dest;
    }
    goto LABEL_15;
  }
  if ( dest == v16 )
  {
    dest = v17;
    v15 = n;
    v16[0] = src[0];
  }
  else
  {
    v9 = v16[0];
    dest = v17;
    v15 = n;
    v16[0] = src[0];
    if ( v8 )
    {
      v17 = v8;
      src[0] = v9;
      goto LABEL_6;
    }
  }
  v17 = src;
  v8 = src;
LABEL_6:
  n = 0;
  *(_BYTE *)v8 = 0;
  if ( v17 != src )
    j_j___libc_free_0(v17, src[0] + 1LL);
  sub_2240AE0(a1 + 136, &dest);
  v11 = *(_QWORD *)(a1 + 240) == 0;
  *(_WORD *)(a1 + 14) = a2;
  if ( v11 )
    sub_4263D6(a1 + 136, &dest, v10);
  (*(void (__fastcall **)(__int64, void **))(a1 + 248))(a1 + 224, &dest);
  if ( dest != v16 )
    j_j___libc_free_0(dest, v16[0] + 1LL);
  return 0;
}
