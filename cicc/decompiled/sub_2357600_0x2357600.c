// Function: sub_2357600
// Address: 0x2357600
//
void __fastcall sub_2357600(unsigned __int64 *a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // r9
  __int64 v4; // r10
  __m128i v5; // xmm0
  __m128i v6; // xmm1
  __m128i v7; // xmm2
  __m128i v8; // xmm3
  __m128i v9; // xmm4
  unsigned __int64 v10; // rax
  _QWORD *v11; // rbx
  unsigned __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdx
  _QWORD *v15; // r12
  unsigned __int64 v16; // r15
  _QWORD *v17; // r13
  __int64 v18; // rax
  __m128i v19; // xmm5
  __m128i v20; // xmm6
  __m128i v21; // xmm7
  int v22; // esi
  __m128i v23; // xmm0
  __m128i v24; // xmm1
  __int64 v25; // rsi
  _QWORD *i; // r14
  _QWORD *j; // r13
  _QWORD *k; // r12
  __int64 v29; // [rsp+0h] [rbp-150h]
  __int64 v30; // [rsp+8h] [rbp-148h]
  __int64 v31; // [rsp+10h] [rbp-140h]
  __int64 v32; // [rsp+18h] [rbp-138h]
  unsigned __int64 v33; // [rsp+20h] [rbp-130h]
  unsigned __int64 v34; // [rsp+28h] [rbp-128h]
  __int64 v35; // [rsp+38h] [rbp-118h] BYREF
  __m128i v36; // [rsp+40h] [rbp-110h] BYREF
  __m128i v37; // [rsp+50h] [rbp-100h] BYREF
  __m128i v38; // [rsp+60h] [rbp-F0h] BYREF
  __m128i v39; // [rsp+70h] [rbp-E0h] BYREF
  __m128i v40; // [rsp+80h] [rbp-D0h] BYREF
  int v41; // [rsp+90h] [rbp-C0h]
  __int64 v42; // [rsp+94h] [rbp-BCh]

  v2 = *(_DWORD *)(a2 + 80);
  v3 = *(_QWORD *)(a2 + 120);
  *(_QWORD *)(a2 + 120) = 0;
  v4 = *(_QWORD *)(a2 + 92);
  v5 = _mm_loadu_si128((const __m128i *)a2);
  v6 = _mm_loadu_si128((const __m128i *)(a2 + 16));
  v7 = _mm_loadu_si128((const __m128i *)(a2 + 32));
  v41 = v2;
  v8 = _mm_loadu_si128((const __m128i *)(a2 + 48));
  v29 = v4;
  v9 = _mm_loadu_si128((const __m128i *)(a2 + 64));
  v30 = v3;
  v42 = *(_QWORD *)(a2 + 84);
  v10 = *(_QWORD *)(a2 + 104);
  v11 = *(_QWORD **)(a2 + 112);
  *(_QWORD *)(a2 + 104) = 0;
  v33 = v10;
  v12 = *(_QWORD *)(a2 + 144);
  *(_QWORD *)(a2 + 112) = 0;
  v36 = v5;
  v37 = v6;
  v38 = v7;
  v39 = v8;
  v40 = v9;
  v34 = v12;
  v13 = *(_QWORD *)(a2 + 160);
  v14 = *(_QWORD *)(a2 + 200);
  v15 = *(_QWORD **)(a2 + 152);
  *(_QWORD *)(a2 + 160) = 0;
  v16 = *(_QWORD *)(a2 + 184);
  v17 = *(_QWORD **)(a2 + 192);
  *(_QWORD *)(a2 + 152) = 0;
  *(_QWORD *)(a2 + 144) = 0;
  *(_QWORD *)(a2 + 200) = 0;
  *(_QWORD *)(a2 + 192) = 0;
  *(_QWORD *)(a2 + 184) = 0;
  v31 = v13;
  v32 = v14;
  v18 = sub_22077B0(0xE8u);
  if ( v18 )
  {
    *(_QWORD *)(v18 + 120) = v11;
    v19 = _mm_loadu_si128(&v36);
    *(_QWORD *)(v18 + 160) = v15;
    v15 = 0;
    v20 = _mm_loadu_si128(&v37);
    v21 = _mm_loadu_si128(&v38);
    *(_QWORD *)(v18 + 192) = v16;
    *(_QWORD *)v18 = &unk_4A0D4B8;
    v22 = v41;
    v16 = 0;
    v23 = _mm_loadu_si128(&v39);
    v24 = _mm_loadu_si128(&v40);
    *(_QWORD *)(v18 + 100) = v29;
    *(_DWORD *)(v18 + 88) = v22;
    v25 = v42;
    *(_QWORD *)(v18 + 152) = v34;
    v11 = 0;
    *(_QWORD *)(v18 + 92) = v25;
    *(_QWORD *)(v18 + 200) = v17;
    v17 = 0;
    *(_QWORD *)(v18 + 112) = v33;
    *(_QWORD *)(v18 + 128) = v30;
    *(_QWORD *)(v18 + 136) = 0;
    *(_QWORD *)(v18 + 144) = 0;
    *(_QWORD *)(v18 + 168) = v31;
    *(_QWORD *)(v18 + 176) = 0;
    *(_QWORD *)(v18 + 184) = 0;
    *(_QWORD *)(v18 + 208) = v32;
    *(__m128i *)(v18 + 8) = v19;
    *(__m128i *)(v18 + 24) = v20;
    *(__m128i *)(v18 + 40) = v21;
    *(__m128i *)(v18 + 56) = v23;
    *(__m128i *)(v18 + 72) = v24;
    *(_QWORD *)(v18 + 216) = 0;
    *(_QWORD *)(v18 + 224) = 0;
    v34 = 0;
    v33 = 0;
  }
  v35 = v18;
  sub_2356EF0(a1, (unsigned __int64 *)&v35);
  if ( v35 )
    (*(void (__fastcall **)(__int64))(*(_QWORD *)v35 + 8LL))(v35);
  for ( i = (_QWORD *)v16; v17 != i; ++i )
  {
    if ( *i )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*i + 8LL))(*i);
  }
  if ( v16 )
    j_j___libc_free_0(v16);
  for ( j = (_QWORD *)v34; j != v15; ++j )
  {
    if ( *j )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*j + 8LL))(*j);
  }
  if ( v34 )
    j_j___libc_free_0(v34);
  for ( k = (_QWORD *)v33; v11 != k; ++k )
  {
    if ( *k )
      (*(void (__fastcall **)(_QWORD))(*(_QWORD *)*k + 8LL))(*k);
  }
  if ( v33 )
    j_j___libc_free_0(v33);
}
