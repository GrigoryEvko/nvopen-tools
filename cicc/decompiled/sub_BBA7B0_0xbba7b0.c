// Function: sub_BBA7B0
// Address: 0xbba7b0
//
__int64 __fastcall sub_BBA7B0(__int64 a1, int a2, __int64 a3, __int64 a4, _BYTE *a5, __int64 a6)
{
  bool v8; // zf
  _BYTE *v9; // rdi
  size_t v10; // rdx
  _QWORD *v11; // rax
  __int64 v12; // rdi
  __int64 *v13; // rdi
  __int64 *v14; // rdi
  int v15; // eax
  _BYTE *v16; // rsi
  __int64 v18; // rax
  _QWORD *v19; // r13
  _QWORD *v20; // r12
  _QWORD *v21; // [rsp+8h] [rbp-88h]
  int v23; // [rsp+1Ch] [rbp-74h] BYREF
  void *dest; // [rsp+20h] [rbp-70h] BYREF
  size_t v25; // [rsp+28h] [rbp-68h]
  _QWORD v26[2]; // [rsp+30h] [rbp-60h] BYREF
  _QWORD *v27; // [rsp+40h] [rbp-50h] BYREF
  size_t n; // [rsp+48h] [rbp-48h]
  _QWORD src[8]; // [rsp+50h] [rbp-40h] BYREF

  v8 = *(_BYTE *)(a1 + 184) == 0;
  v23 = a2;
  dest = v26;
  v25 = 0;
  LOBYTE(v26[0]) = 0;
  if ( !v8 )
  {
    v18 = *(_QWORD *)(a1 + 192);
    if ( v18 != *(_QWORD *)(a1 + 200) )
      *(_QWORD *)(a1 + 200) = v18;
    v19 = *(_QWORD **)(a1 + 144);
    v21 = *(_QWORD **)(a1 + 136);
    if ( v21 != v19 )
    {
      v20 = *(_QWORD **)(a1 + 136);
      do
      {
        if ( (_QWORD *)*v20 != v20 + 2 )
          j_j___libc_free_0(*v20, v20[2] + 1LL);
        v20 += 4;
      }
      while ( v19 != v20 );
      *(_QWORD *)(a1 + 144) = v21;
    }
    *(_BYTE *)(a1 + 184) = 0;
  }
  if ( !a5 )
  {
    LOBYTE(src[0]) = 0;
    v9 = dest;
    v10 = 0;
    v27 = src;
LABEL_24:
    v25 = v10;
    v9[v10] = 0;
    v11 = v27;
    goto LABEL_7;
  }
  v27 = src;
  sub_BB88C0((__int64 *)&v27, a5, (__int64)&a5[a6]);
  v9 = dest;
  v10 = (size_t)v27;
  v11 = dest;
  if ( v27 == src )
  {
    v10 = n;
    if ( n )
    {
      if ( n == 1 )
        *(_BYTE *)dest = src[0];
      else
        memcpy(dest, src, n);
      v10 = n;
      v9 = dest;
    }
    goto LABEL_24;
  }
  a4 = src[0];
  if ( dest == v26 )
  {
    dest = v27;
    v25 = n;
    v26[0] = src[0];
  }
  else
  {
    v12 = v26[0];
    dest = v27;
    v25 = n;
    v26[0] = src[0];
    if ( v11 )
    {
      v27 = v11;
      src[0] = v12;
      goto LABEL_7;
    }
  }
  v27 = src;
  v11 = src;
LABEL_7:
  n = 0;
  *(_BYTE *)v11 = 0;
  if ( v27 != src )
    j_j___libc_free_0(v27, src[0] + 1LL);
  v13 = *(__int64 **)(a1 + 144);
  if ( v13 == *(__int64 **)(a1 + 152) )
  {
    v14 = (__int64 *)(a1 + 136);
    sub_8FD760((__m128i **)(a1 + 136), *(const __m128i **)(a1 + 144), (__int64)&dest);
  }
  else
  {
    if ( v13 )
    {
      *v13 = (__int64)(v13 + 2);
      sub_BB8750(v13, dest, (__int64)dest + v25);
      v13 = *(__int64 **)(a1 + 144);
    }
    v14 = v13 + 4;
    *(_QWORD *)(a1 + 144) = v14;
  }
  v15 = v23;
  v16 = *(_BYTE **)(a1 + 200);
  *(_WORD *)(a1 + 14) = v23;
  if ( v16 == *(_BYTE **)(a1 + 208) )
  {
    v14 = (__int64 *)(a1 + 192);
    sub_B8BBF0(a1 + 192, v16, &v23);
  }
  else
  {
    if ( v16 )
    {
      *(_DWORD *)v16 = v15;
      v16 = *(_BYTE **)(a1 + 200);
    }
    v16 += 4;
    *(_QWORD *)(a1 + 200) = v16;
  }
  if ( !*(_QWORD *)(a1 + 240) )
    sub_4263D6(v14, v16, v10);
  (*(void (__fastcall **)(__int64, void **, size_t, __int64))(a1 + 248))(a1 + 224, &dest, v10, a4);
  if ( dest != v26 )
    j_j___libc_free_0(dest, v26[0] + 1LL);
  return 0;
}
