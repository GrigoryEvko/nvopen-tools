// Function: sub_C63230
// Address: 0xc63230
//
__int64 __fastcall sub_C63230(__int64 a1, int a2, __int64 a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  bool v7; // zf
  __int64 v8; // rax
  _BYTE *v9; // rdi
  _QWORD *v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 v13; // rdx
  int v14; // eax
  _BYTE *v15; // rsi
  size_t v17; // rdx
  int v18; // [rsp+Ch] [rbp-64h] BYREF
  void *dest; // [rsp+10h] [rbp-60h] BYREF
  size_t v20; // [rsp+18h] [rbp-58h]
  _QWORD v21[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v22; // [rsp+30h] [rbp-40h] BYREF
  size_t n; // [rsp+38h] [rbp-38h]
  _QWORD src[6]; // [rsp+40h] [rbp-30h] BYREF

  v7 = *(_BYTE *)(a1 + 168) == 0;
  v18 = a2;
  dest = v21;
  v20 = 0;
  LOBYTE(v21[0]) = 0;
  if ( !v7 )
  {
    v8 = *(_QWORD *)(a1 + 176);
    if ( v8 != *(_QWORD *)(a1 + 184) )
      *(_QWORD *)(a1 + 184) = v8;
    *(_BYTE *)(a1 + 168) = 0;
  }
  if ( !a5 )
  {
    LOBYTE(src[0]) = 0;
    v9 = dest;
    v17 = 0;
    v22 = src;
LABEL_23:
    v20 = v17;
    v9[v17] = 0;
    v10 = v22;
    goto LABEL_10;
  }
  v22 = src;
  sub_C5F830((__int64 *)&v22, a5, (__int64)&a5[a6]);
  v9 = dest;
  v10 = dest;
  if ( v22 == src )
  {
    v17 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v17 = n;
      v9 = dest;
    }
    goto LABEL_23;
  }
  if ( dest == v21 )
  {
    dest = v22;
    v20 = n;
    v21[0] = src[0];
  }
  else
  {
    v11 = v21[0];
    dest = v22;
    v20 = n;
    v21[0] = src[0];
    if ( v10 )
    {
      v22 = v10;
      src[0] = v11;
      goto LABEL_10;
    }
  }
  v22 = src;
  v10 = src;
LABEL_10:
  n = 0;
  *(_BYTE *)v10 = 0;
  if ( v22 != src )
    j_j___libc_free_0(v22, src[0] + 1LL);
  v12 = *(_QWORD *)(a1 + 136);
  sub_C62320(v12, (char **)&dest);
  v14 = v18;
  v15 = *(_BYTE **)(a1 + 184);
  *(_WORD *)(a1 + 14) = v18;
  if ( v15 == *(_BYTE **)(a1 + 192) )
  {
    v12 = a1 + 176;
    sub_B8BBF0(a1 + 176, v15, &v18);
  }
  else
  {
    if ( v15 )
    {
      *(_DWORD *)v15 = v14;
      v15 = *(_BYTE **)(a1 + 184);
    }
    v15 += 4;
    *(_QWORD *)(a1 + 184) = v15;
  }
  if ( !*(_QWORD *)(a1 + 224) )
    sub_4263D6(v12, v15, v13);
  (*(void (__fastcall **)(__int64, void **))(a1 + 232))(a1 + 208, &dest);
  if ( dest != v21 )
    j_j___libc_free_0(dest, v21[0] + 1LL);
  return 0;
}
