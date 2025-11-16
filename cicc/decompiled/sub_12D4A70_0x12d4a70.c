// Function: sub_12D4A70
// Address: 0x12d4a70
//
__int64 __fastcall sub_12D4A70(__int64 a1, int a2, __int64 a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  _QWORD *v7; // rdi
  size_t v8; // rdx
  _QWORD *v9; // rax
  __int64 v10; // rdi
  _QWORD *v11; // rdi
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
    v8 = 0;
    v7 = v15;
    v16 = src;
LABEL_14:
    v14 = v8;
    *((_BYTE *)v7 + v8) = 0;
    v9 = v16;
    goto LABEL_6;
  }
  v16 = src;
  sub_12D3E60((__int64 *)&v16, a5, (__int64)&a5[a6]);
  v7 = dest;
  v8 = (size_t)v16;
  v9 = dest;
  if ( v16 == src )
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
    goto LABEL_14;
  }
  a4 = src[0];
  if ( dest == v15 )
  {
    dest = v16;
    v14 = n;
    v15[0] = src[0];
  }
  else
  {
    v10 = v15[0];
    dest = v16;
    v14 = n;
    v15[0] = src[0];
    if ( v9 )
    {
      v16 = v9;
      src[0] = v10;
      goto LABEL_6;
    }
  }
  v16 = src;
  v9 = src;
LABEL_6:
  n = 0;
  *(_BYTE *)v9 = 0;
  if ( v16 != src )
    j_j___libc_free_0(v16, src[0] + 1LL);
  sub_16C65C0(a1 + 160, &dest, v8, a4);
  v11 = dest;
  *(_DWORD *)(a1 + 16) = a2;
  if ( v11 != v15 )
    j_j___libc_free_0(v11, v15[0] + 1LL);
  return 0;
}
