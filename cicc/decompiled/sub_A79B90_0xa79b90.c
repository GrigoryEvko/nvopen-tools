// Function: sub_A79B90
// Address: 0xa79b90
//
__int64 __fastcall sub_A79B90(__int64 *a1, const void *a2, __int64 a3)
{
  size_t v3; // r14
  __int64 v5; // rbx
  _BYTE *v6; // r8
  unsigned __int64 v7; // rdx
  _BYTE *v8; // rsi
  __int64 v9; // r13
  _BYTE *v11; // rdi
  _BYTE *v12; // [rsp+0h] [rbp-80h] BYREF
  __int64 v13; // [rsp+8h] [rbp-78h]
  _BYTE base[112]; // [rsp+10h] [rbp-70h] BYREF

  v3 = 8 * a3;
  v5 = (8 * a3) >> 3;
  v12 = base;
  v13 = 0x800000000LL;
  if ( (unsigned __int64)(8 * a3) > 0x40 )
  {
    sub_C8D5F0(&v12, base, (8 * a3) >> 3, 8);
    v11 = &v12[8 * (unsigned int)v13];
  }
  else
  {
    v6 = base;
    if ( !v3 )
      goto LABEL_3;
    v11 = base;
  }
  memcpy(v11, a2, v3);
  v6 = v12;
  LODWORD(v3) = v13;
LABEL_3:
  LODWORD(v13) = v5 + v3;
  v7 = (unsigned int)(v5 + v3);
  if ( v7 > 1 )
  {
    qsort(v6, (__int64)(8 * v7) >> 3, 8u, (__compar_fn_t)sub_A73120);
    v6 = v12;
    v7 = (unsigned int)v13;
  }
  v8 = v6;
  v9 = sub_A79A50(a1, (unsigned __int64 *)v6, v7);
  if ( v12 != base )
    _libc_free(v12, v8);
  return v9;
}
