// Function: sub_16B0BF0
// Address: 0x16b0bf0
//
__int64 __fastcall sub_16B0BF0(__int64 a1, int a2, __int64 a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  _QWORD *v7; // rdi
  _QWORD *v8; // rax
  __int64 v9; // rdi
  _QWORD *v10; // rdi
  size_t v12; // rdx
  void *dest; // [rsp+0h] [rbp-60h] BYREF
  size_t v14; // [rsp+8h] [rbp-58h]
  _QWORD v15[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD *v16; // [rsp+20h] [rbp-40h] BYREF
  size_t n; // [rsp+28h] [rbp-38h]
  _QWORD src[6]; // [rsp+30h] [rbp-30h] BYREF

  dest = v15;
  v14 = 0;
  LOBYTE(v15[0]) = 0;
  if ( !a5 )
  {
    LOBYTE(src[0]) = 0;
    v12 = 0;
    v7 = v15;
    v16 = src;
LABEL_14:
    v14 = v12;
    *((_BYTE *)v7 + v12) = 0;
    v8 = v16;
    goto LABEL_6;
  }
  v16 = src;
  sub_16B0480((__int64 *)&v16, a5, (__int64)&a5[a6]);
  v7 = dest;
  v8 = dest;
  if ( v16 == src )
  {
    v12 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v12 = n;
      v7 = dest;
    }
    goto LABEL_14;
  }
  if ( dest == v15 )
  {
    dest = v16;
    v14 = n;
    v15[0] = src[0];
  }
  else
  {
    v9 = v15[0];
    dest = v16;
    v14 = n;
    v15[0] = src[0];
    if ( v8 )
    {
      v16 = v8;
      src[0] = v9;
      goto LABEL_6;
    }
  }
  v16 = src;
  v8 = src;
LABEL_6:
  n = 0;
  *(_BYTE *)v8 = 0;
  if ( v16 != src )
    j_j___libc_free_0(v16, src[0] + 1LL);
  sub_2240AE0(a1 + 160, &dest);
  v10 = dest;
  *(_DWORD *)(a1 + 16) = a2;
  if ( v10 != v15 )
    j_j___libc_free_0(v10, v15[0] + 1LL);
  return 0;
}
