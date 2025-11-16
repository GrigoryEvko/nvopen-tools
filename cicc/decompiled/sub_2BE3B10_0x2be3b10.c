// Function: sub_2BE3B10
// Address: 0x2be3b10
//
void __fastcall sub_2BE3B10(_QWORD *a1)
{
  unsigned __int8 *v1; // rsi
  char v2; // al
  int v3; // eax
  __int64 v4; // rax
  __int64 v5; // r8
  unsigned __int64 v6; // r15
  unsigned __int64 v7; // r12
  unsigned __int64 *v8; // rbx
  unsigned __int64 v9; // r14
  __m128i v10; // xmm0
  __m128i v11; // xmm1
  __int64 v12; // rax
  __m128i v13; // xmm2
  __m128i v14; // xmm3
  int v15; // edx
  unsigned __int64 v16; // rax
  __int64 v17; // rdx
  unsigned __int64 *i; // r14
  unsigned __int64 *v19; // rbx
  unsigned __int64 *v20; // r12
  unsigned __int64 *v21; // [rsp+8h] [rbp-208h]
  __int64 v22; // [rsp+10h] [rbp-200h]
  __int64 v23; // [rsp+18h] [rbp-1F8h]
  __int64 v24; // [rsp+20h] [rbp-1F0h]
  __int64 v25; // [rsp+28h] [rbp-1E8h]
  __int64 v26; // [rsp+30h] [rbp-1E0h]
  __int64 v27; // [rsp+38h] [rbp-1D8h]
  char v28; // [rsp+40h] [rbp-1D0h]
  _QWORD *v29; // [rsp+48h] [rbp-1C8h]
  __int64 v30; // [rsp+50h] [rbp-1C0h]
  unsigned __int64 v31; // [rsp+58h] [rbp-1B8h]
  __m128i v32; // [rsp+60h] [rbp-1B0h] BYREF
  unsigned __int64 v33; // [rsp+70h] [rbp-1A0h]
  __m128i v34; // [rsp+80h] [rbp-190h] BYREF
  __int64 (__fastcall *v35)(unsigned __int64 **, unsigned __int64 **, int); // [rsp+90h] [rbp-180h]
  bool (__fastcall *v36)(_QWORD *, _BYTE *); // [rsp+98h] [rbp-178h]
  unsigned __int64 v37; // [rsp+A0h] [rbp-170h] BYREF
  __int64 v38; // [rsp+A8h] [rbp-168h]
  __int64 v39; // [rsp+B0h] [rbp-160h]
  unsigned __int64 *v40; // [rsp+B8h] [rbp-158h]
  unsigned __int64 *v41; // [rsp+C0h] [rbp-150h]
  __int64 v42; // [rsp+C8h] [rbp-148h]
  unsigned __int64 v43; // [rsp+D0h] [rbp-140h]
  __int64 v44; // [rsp+D8h] [rbp-138h]
  __int64 v45; // [rsp+E0h] [rbp-130h]
  unsigned __int64 v46; // [rsp+E8h] [rbp-128h]
  __int64 v47; // [rsp+F0h] [rbp-120h]
  __int64 v48; // [rsp+F8h] [rbp-118h]
  __int64 v49; // [rsp+100h] [rbp-110h]
  _QWORD *v50; // [rsp+108h] [rbp-108h]
  char v51; // [rsp+110h] [rbp-100h]
  __m128i v52; // [rsp+118h] [rbp-F8h] BYREF
  __m128i v53[7]; // [rsp+128h] [rbp-E8h] BYREF
  int v54; // [rsp+1A0h] [rbp-70h]
  __m128i v55; // [rsp+1B8h] [rbp-58h] BYREF
  __m128i v56[4]; // [rsp+1C8h] [rbp-48h] BYREF

  v1 = (unsigned __int8 *)a1[34];
  v2 = *(_BYTE *)(*(_QWORD *)(a1[49] + 48LL) + 2LL * *v1 + 1);
  v50 = (_QWORD *)a1[48];
  v37 = 0;
  v38 = 0;
  v51 = v2 & 1;
  v39 = 0;
  v40 = 0;
  v41 = 0;
  v42 = 0;
  v43 = 0;
  v44 = 0;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v52 = 0u;
  v53[0] = 0u;
  v3 = sub_2BE10D0(v50, (__int64)v1, (char *)&v1[a1[35]], 0);
  if ( (v3 & 0x10000) == 0 && !(_WORD)v3 )
    abort();
  LOWORD(v49) = v3 | v49;
  BYTE2(v49) |= BYTE2(v3);
  sub_2BE3840((__int64)&v37);
  v4 = v38;
  v5 = v39;
  v38 = 0;
  v39 = 0;
  v30 = v4;
  v22 = v5;
  v23 = v42;
  v21 = (unsigned __int64 *)a1[32];
  v6 = v37;
  v24 = v44;
  v7 = (unsigned __int64)v40;
  v25 = v45;
  v8 = v41;
  v26 = v47;
  v9 = v43;
  v27 = v48;
  v31 = v46;
  v37 = 0;
  v42 = 0;
  v41 = 0;
  v40 = 0;
  v45 = 0;
  v44 = 0;
  v43 = 0;
  v48 = 0;
  v47 = 0;
  v10 = _mm_loadu_si128(&v52);
  v11 = _mm_loadu_si128(v53);
  v46 = 0;
  v54 = v49;
  v35 = 0;
  v29 = v50;
  v55 = v10;
  v28 = v51;
  v56[0] = v11;
  v12 = sub_22077B0(0x98u);
  if ( v12 )
  {
    v13 = _mm_loadu_si128(&v55);
    *(_QWORD *)(v12 + 64) = v25;
    v14 = _mm_loadu_si128(v56);
    *(_QWORD *)(v12 + 40) = v23;
    v15 = v54;
    *(_QWORD *)(v12 + 8) = v30;
    *(_QWORD *)(v12 + 72) = v31;
    *(_QWORD *)(v12 + 16) = v22;
    *(_QWORD *)(v12 + 104) = v29;
    *(_QWORD *)(v12 + 56) = v24;
    *(_QWORD *)(v12 + 80) = v26;
    *(_QWORD *)(v12 + 88) = v27;
    *(_DWORD *)(v12 + 96) = v15;
    *(_BYTE *)(v12 + 112) = v28;
    v31 = 0;
    *(_QWORD *)v12 = v6;
    v6 = 0;
    *(_QWORD *)(v12 + 24) = v7;
    v7 = 0;
    *(_QWORD *)(v12 + 32) = v8;
    v8 = 0;
    *(_QWORD *)(v12 + 48) = v9;
    v9 = 0;
    *(__m128i *)(v12 + 120) = v13;
    *(__m128i *)(v12 + 136) = v14;
  }
  v34.m128i_i64[0] = v12;
  v36 = sub_2BDB730;
  v35 = sub_2BDD2E0;
  v16 = sub_2BE0EB0(v21, &v34);
  v17 = a1[32];
  v32.m128i_i64[1] = v16;
  v33 = v16;
  v32.m128i_i64[0] = v17;
  sub_2BE3490(a1 + 38, &v32);
  if ( v35 )
    v35((unsigned __int64 **)&v34, (unsigned __int64 **)&v34, 3);
  if ( v31 )
    j_j___libc_free_0(v31);
  if ( v9 )
    j_j___libc_free_0(v9);
  for ( i = (unsigned __int64 *)v7; v8 != i; i += 4 )
  {
    if ( (unsigned __int64 *)*i != i + 2 )
      j_j___libc_free_0(*i);
  }
  if ( v7 )
    j_j___libc_free_0(v7);
  if ( v6 )
    j_j___libc_free_0(v6);
  if ( v46 )
    j_j___libc_free_0(v46);
  if ( v43 )
    j_j___libc_free_0(v43);
  v19 = v41;
  v20 = v40;
  if ( v41 != v40 )
  {
    do
    {
      if ( (unsigned __int64 *)*v20 != v20 + 2 )
        j_j___libc_free_0(*v20);
      v20 += 4;
    }
    while ( v19 != v20 );
    v20 = v40;
  }
  if ( v20 )
    j_j___libc_free_0((unsigned __int64)v20);
  if ( v37 )
    j_j___libc_free_0(v37);
}
