// Function: sub_2BE4580
// Address: 0x2be4580
//
void __fastcall sub_2BE4580(_QWORD *a1)
{
  unsigned __int8 *v1; // rsi
  char v2; // al
  int v3; // eax
  unsigned __int64 v4; // rbx
  char *v5; // rax
  __int64 v6; // rax
  char v7; // al
  char v8; // r12
  char v9; // r8
  unsigned __int64 v10; // rdx
  __int64 v11; // rax
  char *v12; // r11
  __int64 v13; // rdx
  unsigned __int64 v14; // r14
  unsigned __int64 *v15; // r15
  unsigned __int64 *v16; // r13
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
  unsigned __int64 v27; // rdi
  unsigned __int64 *j; // rbx
  unsigned __int64 *v29; // rbx
  unsigned __int64 *v30; // r12
  unsigned __int64 v31; // rdi
  unsigned __int64 *v32; // rbx
  unsigned __int64 *v33; // r12
  unsigned __int64 *v34; // [rsp+8h] [rbp-208h]
  char *v35; // [rsp+10h] [rbp-200h]
  __int64 v36; // [rsp+18h] [rbp-1F8h]
  __int64 v37; // [rsp+20h] [rbp-1F0h]
  __int64 v38; // [rsp+28h] [rbp-1E8h]
  __int64 v39; // [rsp+30h] [rbp-1E0h]
  __int64 v40; // [rsp+38h] [rbp-1D8h]
  unsigned __int64 v41; // [rsp+48h] [rbp-1C8h]
  char *v42; // [rsp+50h] [rbp-1C0h]
  unsigned __int64 v43; // [rsp+58h] [rbp-1B8h]
  __m128i v44; // [rsp+60h] [rbp-1B0h] BYREF
  __int64 v45; // [rsp+70h] [rbp-1A0h]
  __m128i v46; // [rsp+80h] [rbp-190h] BYREF
  __int64 (__fastcall *v47)(unsigned __int64 **, const __m128i **, int); // [rsp+90h] [rbp-180h]
  bool (__fastcall *v48)(_QWORD *, _BYTE *); // [rsp+98h] [rbp-178h]
  char *v49; // [rsp+A0h] [rbp-170h] BYREF
  char *v50; // [rsp+A8h] [rbp-168h]
  __int64 v51; // [rsp+B0h] [rbp-160h]
  unsigned __int64 *v52; // [rsp+B8h] [rbp-158h]
  unsigned __int64 *v53; // [rsp+C0h] [rbp-150h]
  __int64 v54; // [rsp+C8h] [rbp-148h]
  unsigned __int64 *v55; // [rsp+D0h] [rbp-140h]
  unsigned __int64 *v56; // [rsp+D8h] [rbp-138h]
  __int64 v57; // [rsp+E0h] [rbp-130h]
  unsigned __int64 v58; // [rsp+E8h] [rbp-128h]
  __int64 v59; // [rsp+F0h] [rbp-120h]
  __int64 v60; // [rsp+F8h] [rbp-118h]
  __int64 v61; // [rsp+100h] [rbp-110h]
  _QWORD *v62; // [rsp+108h] [rbp-108h]
  _QWORD *v63; // [rsp+110h] [rbp-100h]
  char v64; // [rsp+118h] [rbp-F8h]
  __m128i v65; // [rsp+120h] [rbp-F0h] BYREF
  __m128i v66; // [rsp+130h] [rbp-E0h] BYREF
  char **v67; // [rsp+140h] [rbp-D0h] BYREF
  char v68; // [rsp+148h] [rbp-C8h]
  int v69; // [rsp+1A0h] [rbp-70h]
  _QWORD *v70; // [rsp+1A8h] [rbp-68h]
  __m128i v71; // [rsp+1C0h] [rbp-50h] BYREF
  __m128i v72[4]; // [rsp+1D0h] [rbp-40h] BYREF

  v1 = (unsigned __int8 *)a1[34];
  v2 = *(_BYTE *)(*(_QWORD *)(a1[49] + 48LL) + 2LL * *v1 + 1);
  v62 = (_QWORD *)a1[48];
  v63 = v62;
  v49 = 0;
  v64 = v2 & 1;
  v50 = 0;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v65 = 0u;
  v66 = 0u;
  v3 = sub_2BE10D0(v62, (__int64)v1, (char *)&v1[a1[35]], 0);
  if ( (v3 & 0x10000) == 0 && !(_WORD)v3 )
    abort();
  LOWORD(v61) = v3 | v61;
  BYTE2(v61) |= BYTE2(v3);
  v4 = 0;
  v5 = sub_2BDBC30(v49, v49);
  sub_2BE3780((__int64)&v49, v5, v50);
  do
  {
    v67 = &v49;
    v68 = v4;
    v7 = sub_2BE41C0((char *)&v67);
    v8 = v64;
    v9 = v7;
    v10 = v4 >> 6;
    v11 = 1LL << v4;
    if ( v9 == v64 )
      v6 = v65.m128i_i64[v10] & ~v11;
    else
      v6 = v65.m128i_i64[v10] | v11;
    ++v4;
    v65.m128i_i64[v10] = v6;
  }
  while ( v4 != 256 );
  v12 = v50;
  v50 = 0;
  v13 = v51;
  v51 = 0;
  v35 = v12;
  v36 = v13;
  v42 = v49;
  v34 = (unsigned __int64 *)a1[32];
  v43 = (unsigned __int64)v55;
  v37 = v54;
  v14 = (unsigned __int64)v52;
  v38 = v57;
  v15 = v53;
  v39 = v59;
  v16 = v56;
  v41 = v58;
  v49 = 0;
  v54 = 0;
  v53 = 0;
  v52 = 0;
  v57 = 0;
  v56 = 0;
  v55 = 0;
  v40 = v60;
  v17 = _mm_loadu_si128(&v65);
  v18 = _mm_loadu_si128(&v66);
  v60 = 0;
  v69 = v61;
  v19 = v63;
  v71 = v17;
  v59 = 0;
  v58 = 0;
  v70 = v62;
  v47 = 0;
  v72[0] = v18;
  v20 = sub_22077B0(0xA0u);
  if ( v20 )
  {
    v21 = _mm_loadu_si128(&v71);
    *(_QWORD *)(v20 + 16) = v36;
    v22 = v69;
    v23 = _mm_loadu_si128(v72);
    *(_QWORD *)(v20 + 64) = v38;
    *(_QWORD *)v20 = v42;
    *(_DWORD *)(v20 + 96) = v22;
    v24 = v70;
    *(_QWORD *)(v20 + 8) = v35;
    *(_QWORD *)(v20 + 40) = v37;
    *(_QWORD *)(v20 + 48) = v43;
    *(_QWORD *)(v20 + 72) = v41;
    *(_QWORD *)(v20 + 80) = v39;
    *(_QWORD *)(v20 + 88) = v40;
    *(_QWORD *)(v20 + 104) = v24;
    *(_BYTE *)(v20 + 120) = v8;
    v41 = 0;
    v43 = 0;
    v42 = 0;
    *(_QWORD *)(v20 + 24) = v14;
    v14 = 0;
    *(_QWORD *)(v20 + 32) = v15;
    v15 = 0;
    *(_QWORD *)(v20 + 56) = v16;
    v16 = 0;
    *(_QWORD *)(v20 + 112) = v19;
    *(__m128i *)(v20 + 128) = v21;
    *(__m128i *)(v20 + 144) = v23;
  }
  v46.m128i_i64[0] = v20;
  v48 = sub_2BDB750;
  v47 = sub_2BDDB90;
  v44.m128i_i64[1] = sub_2BE0EB0(v34, &v46);
  v25 = a1[32];
  v45 = v44.m128i_i64[1];
  v44.m128i_i64[0] = v25;
  sub_2BE3490(a1 + 38, &v44);
  if ( v47 )
    v47((unsigned __int64 **)&v46, (const __m128i **)&v46, 3);
  if ( v41 )
    j_j___libc_free_0(v41);
  for ( i = (unsigned __int64 *)v43; i != v16; i += 8 )
  {
    v27 = i[4];
    if ( (unsigned __int64 *)v27 != i + 6 )
      j_j___libc_free_0(v27);
    if ( (unsigned __int64 *)*i != i + 2 )
      j_j___libc_free_0(*i);
  }
  if ( v43 )
    j_j___libc_free_0(v43);
  for ( j = (unsigned __int64 *)v14; j != v15; j += 4 )
  {
    if ( (unsigned __int64 *)*j != j + 2 )
      j_j___libc_free_0(*j);
  }
  if ( v14 )
    j_j___libc_free_0(v14);
  if ( v42 )
    j_j___libc_free_0((unsigned __int64)v42);
  if ( v58 )
    j_j___libc_free_0(v58);
  v29 = v56;
  v30 = v55;
  if ( v56 != v55 )
  {
    do
    {
      v31 = v30[4];
      if ( (unsigned __int64 *)v31 != v30 + 6 )
        j_j___libc_free_0(v31);
      if ( (unsigned __int64 *)*v30 != v30 + 2 )
        j_j___libc_free_0(*v30);
      v30 += 8;
    }
    while ( v29 != v30 );
    v30 = v55;
  }
  if ( v30 )
    j_j___libc_free_0((unsigned __int64)v30);
  v32 = v53;
  v33 = v52;
  if ( v53 != v52 )
  {
    do
    {
      if ( (unsigned __int64 *)*v33 != v33 + 2 )
        j_j___libc_free_0(*v33);
      v33 += 4;
    }
    while ( v32 != v33 );
    v33 = v52;
  }
  if ( v33 )
    j_j___libc_free_0((unsigned __int64)v33);
  if ( v49 )
    j_j___libc_free_0((unsigned __int64)v49);
}
