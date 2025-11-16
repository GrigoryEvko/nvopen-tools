// Function: sub_16FA590
// Address: 0x16fa590
//
__int64 __fastcall sub_16FA590(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, int a6)
{
  unsigned __int64 *v6; // r13
  unsigned int v8; // edx
  int v9; // eax
  __int64 *v10; // r9
  __int64 v11; // rax
  __int64 v12; // rax
  __m128i v13; // xmm0
  _BYTE *v14; // rsi
  unsigned __int64 v15; // rbx
  __int64 v16; // rdx
  __int64 v17; // rax
  unsigned __int64 v18; // rdx
  _QWORD *v19; // rdi
  unsigned __int64 *v21; // rbx
  __int64 v22; // rcx
  __int64 v23; // rax
  int v24; // r8d
  __int64 v25; // rax
  _BYTE *v26; // rsi
  __int64 v27; // rdx
  unsigned __int64 v28; // r14
  __m128i v29; // xmm2
  unsigned __int64 v30; // rdx
  unsigned __int64 v31; // rax
  int v32; // r9d
  _QWORD *v33; // rdi
  int v34; // [rsp+4h] [rbp-7Ch]
  __int64 *v35; // [rsp+8h] [rbp-78h]
  __int64 v36; // [rsp+8h] [rbp-78h]
  __m128i v37; // [rsp+18h] [rbp-68h] BYREF
  _QWORD *v38; // [rsp+28h] [rbp-58h]
  __int64 v39; // [rsp+30h] [rbp-50h]
  _QWORD v40[9]; // [rsp+38h] [rbp-48h] BYREF

  v6 = (unsigned __int64 *)(a1 + 184);
  v8 = *(_DWORD *)(a1 + 240);
  if ( v8 )
  {
    v21 = *(unsigned __int64 **)(a1 + 192);
    v22 = *(_QWORD *)(a1 + 232) + 24LL * v8 - 24;
    v23 = *(_QWORD *)v22;
    v24 = *(_DWORD *)(v22 + 8);
    *(_DWORD *)(a1 + 240) = v8 - 1;
    v38 = v40;
    v39 = 0;
    LOBYTE(v40[0]) = 0;
    v37 = _mm_loadu_si128((const __m128i *)(v23 + 24));
    if ( v21 != (unsigned __int64 *)v23 && v21 != v6 )
    {
      do
        v21 = (unsigned __int64 *)v21[1];
      while ( v21 != v6 && (unsigned __int64 *)v23 != v21 );
    }
    v34 = v24;
    v36 = a1 + 80;
    v25 = sub_145CBF0((__int64 *)(a1 + 80), 72, 16);
    v26 = v38;
    v27 = v39;
    v28 = v25;
    *(_QWORD *)v25 = 0;
    v29 = _mm_loadu_si128(&v37);
    *(_QWORD *)(v25 + 8) = 0;
    *(__m128i *)(v25 + 24) = v29;
    *(_DWORD *)(v25 + 16) = 16;
    *(_QWORD *)(v25 + 40) = v25 + 56;
    sub_16F6740((__int64 *)(v25 + 40), v26, (__int64)&v26[v27]);
    v30 = *v21;
    v31 = *(_QWORD *)v28;
    *(_QWORD *)(v28 + 8) = v21;
    v30 &= 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)v28 = v30 | v31 & 7;
    *(_QWORD *)(v30 + 8) = v28;
    *v21 = v28 | *v21 & 7;
    sub_16FA1A0(a1, v34, 10, (unsigned __int64 *)v28, v34, v32);
    v33 = v38;
    *(_BYTE *)(a1 + 73) = 0;
    v10 = (__int64 *)(a1 + 80);
    if ( v33 != v40 )
    {
      j_j___libc_free_0(v33, v40[0] + 1LL);
      v10 = (__int64 *)v36;
    }
  }
  else
  {
    v9 = *(_DWORD *)(a1 + 68);
    if ( !v9 )
    {
      sub_16FA1A0(a1, *(_DWORD *)(a1 + 60), 10, (unsigned __int64 *)(a1 + 184), a5, a6);
      v9 = *(_DWORD *)(a1 + 68);
    }
    v10 = (__int64 *)(a1 + 80);
    *(_BYTE *)(a1 + 73) = v9 == 0;
  }
  v11 = *(_QWORD *)(a1 + 40);
  v35 = v10;
  v38 = v40;
  v37.m128i_i64[0] = v11;
  v39 = 0;
  LOBYTE(v40[0]) = 0;
  v37.m128i_i64[1] = 1;
  sub_16F7930(a1, 1u);
  v12 = sub_145CBF0(v35, 72, 16);
  v13 = _mm_loadu_si128(&v37);
  v14 = v38;
  v15 = v12;
  *(_QWORD *)v12 = 0;
  v16 = v39;
  *(_QWORD *)(v12 + 8) = 0;
  *(__m128i *)(v12 + 24) = v13;
  *(_DWORD *)(v12 + 16) = 17;
  *(_QWORD *)(v12 + 40) = v12 + 56;
  sub_16F6740((__int64 *)(v12 + 40), v14, (__int64)&v14[v16]);
  v17 = *(_QWORD *)v15;
  *(_QWORD *)(v15 + 8) = v6;
  v18 = *(_QWORD *)(a1 + 184) & 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v15 = v18 | v17 & 7;
  *(_QWORD *)(v18 + 8) = v15;
  v19 = v38;
  *(_QWORD *)(a1 + 184) = *(_QWORD *)(a1 + 184) & 7LL | v15;
  if ( v19 != v40 )
    j_j___libc_free_0(v19, v40[0] + 1LL);
  return 1;
}
