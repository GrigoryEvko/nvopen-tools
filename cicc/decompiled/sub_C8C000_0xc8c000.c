// Function: sub_C8C000
// Address: 0xc8c000
//
__int64 __fastcall sub_C8C000(__int64 a1, __int16 a2, __int64 a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  _QWORD *v7; // rdi
  _QWORD *v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // rdx
  bool v12; // zf
  size_t v14; // rdx
  void *dest; // [rsp+0h] [rbp-60h] BYREF
  size_t v16; // [rsp+8h] [rbp-58h]
  _QWORD v17[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD *v18; // [rsp+20h] [rbp-40h] BYREF
  size_t n; // [rsp+28h] [rbp-38h]
  _QWORD src[6]; // [rsp+30h] [rbp-30h] BYREF

  dest = v17;
  v16 = 0;
  LOBYTE(v17[0]) = 0;
  if ( !a5 )
  {
    LOBYTE(src[0]) = 0;
    v14 = 0;
    v7 = v17;
    v18 = src;
LABEL_15:
    v16 = v14;
    *((_BYTE *)v7 + v14) = 0;
    v8 = v18;
    goto LABEL_6;
  }
  v18 = src;
  sub_C8B520((__int64 *)&v18, a5, (__int64)&a5[a6]);
  v7 = dest;
  v8 = dest;
  if ( v18 == src )
  {
    v14 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v14 = n;
      v7 = dest;
    }
    goto LABEL_15;
  }
  if ( dest == v17 )
  {
    dest = v18;
    v16 = n;
    v17[0] = src[0];
  }
  else
  {
    v9 = v17[0];
    dest = v18;
    v16 = n;
    v17[0] = src[0];
    if ( v8 )
    {
      v18 = v8;
      src[0] = v9;
      goto LABEL_6;
    }
  }
  v18 = src;
  v8 = src;
LABEL_6:
  n = 0;
  *(_BYTE *)v8 = 0;
  if ( v18 != src )
    j_j___libc_free_0(v18, src[0] + 1LL);
  v10 = *(_QWORD *)(a1 + 136);
  sub_2240AE0(v10, &dest);
  v12 = *(_QWORD *)(a1 + 216) == 0;
  *(_WORD *)(a1 + 14) = a2;
  if ( v12 )
    sub_4263D6(v10, &dest, v11);
  (*(void (__fastcall **)(__int64, void **))(a1 + 224))(a1 + 200, &dest);
  if ( dest != v17 )
    j_j___libc_free_0(dest, v17[0] + 1LL);
  return 0;
}
