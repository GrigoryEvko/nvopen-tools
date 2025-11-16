// Function: sub_12F3E60
// Address: 0x12f3e60
//
__int64 __fastcall sub_12F3E60(__int64 a1, int a2, __int64 a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  _QWORD *v7; // rdi
  _QWORD *v8; // rax
  __int64 v9; // rdi
  __int64 *v10; // rdi
  int v11; // eax
  _BYTE *v12; // rsi
  size_t v14; // rdx
  int v15; // [rsp+Ch] [rbp-64h] BYREF
  void *dest; // [rsp+10h] [rbp-60h] BYREF
  size_t v17; // [rsp+18h] [rbp-58h]
  _QWORD v18[2]; // [rsp+20h] [rbp-50h] BYREF
  _QWORD *v19; // [rsp+30h] [rbp-40h] BYREF
  size_t n; // [rsp+38h] [rbp-38h]
  _QWORD src[6]; // [rsp+40h] [rbp-30h] BYREF

  v15 = a2;
  dest = v18;
  v17 = 0;
  LOBYTE(v18[0]) = 0;
  if ( !a5 )
  {
    LOBYTE(src[0]) = 0;
    v14 = 0;
    v7 = v18;
    v19 = src;
LABEL_22:
    v17 = v14;
    *((_BYTE *)v7 + v14) = 0;
    v8 = v19;
    goto LABEL_6;
  }
  v19 = src;
  sub_12EFD20((__int64 *)&v19, a5, (__int64)&a5[a6]);
  v7 = dest;
  v8 = dest;
  if ( v19 == src )
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
    goto LABEL_22;
  }
  if ( dest == v18 )
  {
    dest = v19;
    v17 = n;
    v18[0] = src[0];
  }
  else
  {
    v9 = v18[0];
    dest = v19;
    v17 = n;
    v18[0] = src[0];
    if ( v8 )
    {
      v19 = v8;
      src[0] = v9;
      goto LABEL_6;
    }
  }
  v19 = src;
  v8 = src;
LABEL_6:
  n = 0;
  *(_BYTE *)v8 = 0;
  if ( v19 != src )
    j_j___libc_free_0(v19, src[0] + 1LL);
  v10 = *(__int64 **)(a1 + 168);
  if ( v10 == *(__int64 **)(a1 + 176) )
  {
    sub_8FD760((__m128i **)(a1 + 160), *(const __m128i **)(a1 + 168), (__int64)&dest);
  }
  else
  {
    if ( v10 )
    {
      *v10 = (__int64)(v10 + 2);
      sub_12EEFD0(v10, dest, (__int64)dest + v17);
      v10 = *(__int64 **)(a1 + 168);
    }
    *(_QWORD *)(a1 + 168) = v10 + 4;
  }
  v11 = v15;
  v12 = *(_BYTE **)(a1 + 192);
  *(_DWORD *)(a1 + 16) = v15;
  if ( v12 == *(_BYTE **)(a1 + 200) )
  {
    sub_B8BBF0(a1 + 184, v12, &v15);
  }
  else
  {
    if ( v12 )
    {
      *(_DWORD *)v12 = v11;
      v12 = *(_BYTE **)(a1 + 192);
    }
    *(_QWORD *)(a1 + 192) = v12 + 4;
  }
  if ( dest != v18 )
    j_j___libc_free_0(dest, v18[0] + 1LL);
  return 0;
}
