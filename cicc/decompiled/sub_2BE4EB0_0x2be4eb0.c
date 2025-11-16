// Function: sub_2BE4EB0
// Address: 0x2be4eb0
//
void __fastcall sub_2BE4EB0(_QWORD *a1)
{
  unsigned __int8 *v1; // rsi
  char v2; // al
  int v3; // eax
  unsigned __int64 v4; // rbx
  char *v5; // rsi
  __int64 v6; // rax
  char v7; // al
  char v8; // r12
  char v9; // r8
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // r9
  char *v14; // r13
  unsigned __int64 v15; // r14
  unsigned __int64 *v16; // r15
  __m128i v17; // xmm0
  __m128i v18; // xmm1
  _QWORD *v19; // rbx
  __int64 v20; // rax
  __m128i v21; // xmm2
  int v22; // edx
  __m128i v23; // xmm3
  _QWORD *v24; // rdx
  __int64 v25; // rdx
  unsigned __int64 *i; // rbx
  unsigned __int64 *v27; // rbx
  unsigned __int64 *v28; // r12
  unsigned __int64 *v29; // [rsp+8h] [rbp-208h]
  __int64 v30; // [rsp+10h] [rbp-200h]
  __int64 v31; // [rsp+18h] [rbp-1F8h]
  __int64 v32; // [rsp+20h] [rbp-1F0h]
  __int64 v33; // [rsp+28h] [rbp-1E8h]
  __int64 v34; // [rsp+30h] [rbp-1E0h]
  __int64 v35; // [rsp+38h] [rbp-1D8h]
  char *v36; // [rsp+40h] [rbp-1D0h]
  unsigned __int64 v37; // [rsp+50h] [rbp-1C0h]
  unsigned __int64 v38; // [rsp+58h] [rbp-1B8h]
  __m128i v39; // [rsp+60h] [rbp-1B0h] BYREF
  __int64 v40; // [rsp+70h] [rbp-1A0h]
  __m128i v41; // [rsp+80h] [rbp-190h] BYREF
  __int64 (__fastcall *v42)(unsigned __int64 **, const __m128i **, int); // [rsp+90h] [rbp-180h]
  bool (__fastcall *v43)(_QWORD *, _BYTE *); // [rsp+98h] [rbp-178h]
  char *v44; // [rsp+A0h] [rbp-170h] BYREF
  char *v45; // [rsp+A8h] [rbp-168h]
  __int64 v46; // [rsp+B0h] [rbp-160h]
  unsigned __int64 *v47; // [rsp+B8h] [rbp-158h]
  unsigned __int64 *v48; // [rsp+C0h] [rbp-150h]
  __int64 v49; // [rsp+C8h] [rbp-148h]
  unsigned __int64 v50; // [rsp+D0h] [rbp-140h]
  __int64 v51; // [rsp+D8h] [rbp-138h]
  __int64 v52; // [rsp+E0h] [rbp-130h]
  unsigned __int64 v53; // [rsp+E8h] [rbp-128h]
  __int64 v54; // [rsp+F0h] [rbp-120h]
  __int64 v55; // [rsp+F8h] [rbp-118h]
  __int64 v56; // [rsp+100h] [rbp-110h]
  _QWORD *v57; // [rsp+108h] [rbp-108h]
  _QWORD *v58; // [rsp+110h] [rbp-100h]
  char v59; // [rsp+118h] [rbp-F8h]
  __m128i v60; // [rsp+120h] [rbp-F0h] BYREF
  __m128i v61; // [rsp+130h] [rbp-E0h] BYREF
  char **v62; // [rsp+140h] [rbp-D0h] BYREF
  char v63; // [rsp+148h] [rbp-C8h]
  int v64; // [rsp+1A0h] [rbp-70h]
  _QWORD *v65; // [rsp+1A8h] [rbp-68h]
  __m128i v66; // [rsp+1C0h] [rbp-50h] BYREF
  __m128i v67[4]; // [rsp+1D0h] [rbp-40h] BYREF

  v1 = (unsigned __int8 *)a1[34];
  v2 = *(_BYTE *)(*(_QWORD *)(a1[49] + 48LL) + 2LL * *v1 + 1);
  v57 = (_QWORD *)a1[48];
  v58 = v57;
  v44 = 0;
  v59 = v2 & 1;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v60 = 0u;
  v61 = 0u;
  v3 = sub_2BE10D0(v57, (__int64)v1, (char *)&v1[a1[35]], 1);
  if ( (v3 & 0x10000) == 0 && !(_WORD)v3 )
    abort();
  LOWORD(v56) = v3 | v56;
  BYTE2(v56) |= BYTE2(v3);
  v4 = 0;
  v5 = sub_2BDBC30(v44, v44);
  sub_2BE3780((__int64)&v44, v5, v45);
  do
  {
    v62 = &v44;
    v63 = v4;
    v7 = sub_2BE4CE0((char *)&v62, (__int64)v5);
    v8 = v59;
    v9 = v7;
    v10 = v4 >> 6;
    v11 = 1LL << v4;
    if ( v9 == v59 )
      v6 = v60.m128i_i64[v10] & ~v11;
    else
      v6 = v60.m128i_i64[v10] | v11;
    ++v4;
    v60.m128i_i64[v10] = v6;
  }
  while ( v4 != 256 );
  v12 = v46;
  v46 = 0;
  v13 = v49;
  v49 = 0;
  v30 = v12;
  v31 = v13;
  v36 = v45;
  v29 = (unsigned __int64 *)a1[32];
  v38 = v50;
  v32 = v51;
  v14 = v44;
  v33 = v52;
  v15 = (unsigned __int64)v47;
  v34 = v54;
  v16 = v48;
  v37 = v53;
  v45 = 0;
  v44 = 0;
  v48 = 0;
  v47 = 0;
  v52 = 0;
  v51 = 0;
  v50 = 0;
  v35 = v55;
  v17 = _mm_loadu_si128(&v60);
  v18 = _mm_loadu_si128(&v61);
  v55 = 0;
  v64 = v56;
  v19 = v58;
  v66 = v17;
  v54 = 0;
  v53 = 0;
  v65 = v57;
  v42 = 0;
  v67[0] = v18;
  v20 = sub_22077B0(0xA0u);
  if ( v20 )
  {
    v21 = _mm_loadu_si128(&v66);
    *(_QWORD *)(v20 + 16) = v30;
    v22 = v64;
    v23 = _mm_loadu_si128(v67);
    *(_QWORD *)(v20 + 64) = v33;
    *(_QWORD *)(v20 + 8) = v36;
    *(_DWORD *)(v20 + 96) = v22;
    v24 = v65;
    *(_QWORD *)(v20 + 40) = v31;
    *(_QWORD *)(v20 + 48) = v38;
    *(_QWORD *)(v20 + 56) = v32;
    *(_QWORD *)(v20 + 72) = v37;
    *(_QWORD *)(v20 + 80) = v34;
    *(_QWORD *)(v20 + 88) = v35;
    *(_QWORD *)(v20 + 104) = v24;
    *(_BYTE *)(v20 + 120) = v8;
    v37 = 0;
    v38 = 0;
    *(_QWORD *)v20 = v14;
    v14 = 0;
    *(_QWORD *)(v20 + 24) = v15;
    v15 = 0;
    *(_QWORD *)(v20 + 32) = v16;
    v16 = 0;
    *(_QWORD *)(v20 + 112) = v19;
    *(__m128i *)(v20 + 128) = v21;
    *(__m128i *)(v20 + 144) = v23;
  }
  v41.m128i_i64[0] = v20;
  v43 = sub_2BDB780;
  v42 = sub_2BDD6A0;
  v39.m128i_i64[1] = sub_2BE0EB0(v29, &v41);
  v25 = a1[32];
  v40 = v39.m128i_i64[1];
  v39.m128i_i64[0] = v25;
  sub_2BE3490(a1 + 38, &v39);
  if ( v42 )
    v42((unsigned __int64 **)&v41, (const __m128i **)&v41, 3);
  if ( v37 )
    j_j___libc_free_0(v37);
  if ( v38 )
    j_j___libc_free_0(v38);
  for ( i = (unsigned __int64 *)v15; i != v16; i += 4 )
  {
    if ( (unsigned __int64 *)*i != i + 2 )
      j_j___libc_free_0(*i);
  }
  if ( v15 )
    j_j___libc_free_0(v15);
  if ( v14 )
    j_j___libc_free_0((unsigned __int64)v14);
  if ( v53 )
    j_j___libc_free_0(v53);
  if ( v50 )
    j_j___libc_free_0(v50);
  v27 = v48;
  v28 = v47;
  if ( v48 != v47 )
  {
    do
    {
      if ( (unsigned __int64 *)*v28 != v28 + 2 )
        j_j___libc_free_0(*v28);
      v28 += 4;
    }
    while ( v27 != v28 );
    v28 = v47;
  }
  if ( v28 )
    j_j___libc_free_0((unsigned __int64)v28);
  if ( v44 )
    j_j___libc_free_0((unsigned __int64)v44);
}
