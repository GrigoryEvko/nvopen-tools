// Function: sub_226B0F0
// Address: 0x226b0f0
//
__int64 __fastcall sub_226B0F0(__int64 a1, __int16 a2, __int64 a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  _QWORD *v7; // rdi
  size_t v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rdx
  bool v12; // zf
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
    v8 = 0;
    v7 = v16;
    v17 = src;
LABEL_15:
    v15 = v8;
    *((_BYTE *)v7 + v8) = 0;
    v9 = v17;
    goto LABEL_6;
  }
  v17 = src;
  sub_226ACC0((__int64 *)&v17, a5, (__int64)&a5[a6]);
  v7 = dest;
  v8 = (size_t)v17;
  v9 = dest;
  if ( v17 == src )
  {
    v8 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v8 = n;
      v7 = dest;
    }
    goto LABEL_15;
  }
  a4 = src[0];
  if ( dest == v16 )
  {
    dest = v17;
    v15 = n;
    v16[0] = src[0];
  }
  else
  {
    v10 = v16[0];
    dest = v17;
    v15 = n;
    v16[0] = src[0];
    if ( v9 )
    {
      v17 = v9;
      src[0] = v10;
      goto LABEL_6;
    }
  }
  v17 = src;
  v9 = src;
LABEL_6:
  n = 0;
  *(_BYTE *)v9 = 0;
  if ( v17 != src )
    j_j___libc_free_0((unsigned __int64)v17);
  sub_23C6990(a1 + 136, &dest, v8, a4);
  v12 = *(_QWORD *)(a1 + 168) == 0;
  *(_WORD *)(a1 + 14) = a2;
  if ( v12 )
    sub_4263D6(a1 + 136, &dest, v11);
  (*(void (__fastcall **)(__int64, void **))(a1 + 176))(a1 + 152, &dest);
  if ( dest != v16 )
    j_j___libc_free_0((unsigned __int64)dest);
  return 0;
}
