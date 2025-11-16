// Function: sub_2BEB870
// Address: 0x2beb870
//
void __fastcall sub_2BEB870(__int64 a1, char a2)
{
  __int64 v2; // rax
  char *v3; // r12
  char *v4; // r15
  char *v5; // rdi
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rbx
  char *v8; // rax
  __int64 v9; // rax
  char v10; // al
  char v11; // r12
  char v12; // r8
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  char *v15; // r11
  __int64 v16; // rdx
  unsigned __int64 v17; // r14
  unsigned __int64 *v18; // r15
  unsigned __int64 *v19; // r13
  __m128i v20; // xmm0
  __m128i v21; // xmm1
  __int64 v22; // rbx
  __int64 v23; // rax
  __m128i v24; // xmm2
  int v25; // edx
  __m128i v26; // xmm3
  __int64 v27; // rdx
  __int64 v28; // rdx
  unsigned __int64 *i; // rbx
  unsigned __int64 v30; // rdi
  unsigned __int64 *j; // rbx
  unsigned __int64 *v32; // rbx
  unsigned __int64 *v33; // r12
  unsigned __int64 v34; // rdi
  unsigned __int64 *v35; // rbx
  unsigned __int64 *v36; // r12
  _BYTE *v37; // rax
  unsigned __int64 *v38; // [rsp+8h] [rbp-218h]
  char *v39; // [rsp+10h] [rbp-210h]
  __int64 v40; // [rsp+18h] [rbp-208h]
  __int64 v41; // [rsp+20h] [rbp-200h]
  __int64 v42; // [rsp+28h] [rbp-1F8h]
  __int64 v43; // [rsp+30h] [rbp-1F0h]
  __int64 v44; // [rsp+38h] [rbp-1E8h]
  unsigned __int64 v45; // [rsp+40h] [rbp-1E0h]
  unsigned __int64 v46; // [rsp+48h] [rbp-1D8h]
  unsigned __int64 v48; // [rsp+58h] [rbp-1C8h]
  __int16 v49; // [rsp+6Eh] [rbp-1B2h] BYREF
  __m128i v50; // [rsp+70h] [rbp-1B0h] BYREF
  __int64 v51; // [rsp+80h] [rbp-1A0h]
  __m128i v52; // [rsp+90h] [rbp-190h] BYREF
  __int64 (__fastcall *v53)(unsigned __int64 **, const __m128i **, int); // [rsp+A0h] [rbp-180h]
  bool (__fastcall *v54)(_QWORD *, _BYTE *); // [rsp+A8h] [rbp-178h]
  unsigned __int64 v55; // [rsp+B0h] [rbp-170h] BYREF
  char *v56; // [rsp+B8h] [rbp-168h]
  __int64 v57; // [rsp+C0h] [rbp-160h]
  unsigned __int64 *v58; // [rsp+C8h] [rbp-158h]
  unsigned __int64 *v59; // [rsp+D0h] [rbp-150h]
  __int64 v60; // [rsp+D8h] [rbp-148h]
  unsigned __int64 *v61; // [rsp+E0h] [rbp-140h]
  unsigned __int64 *v62; // [rsp+E8h] [rbp-138h]
  __int64 v63; // [rsp+F0h] [rbp-130h]
  unsigned __int64 v64; // [rsp+F8h] [rbp-128h]
  __int64 v65; // [rsp+100h] [rbp-120h]
  __int64 v66; // [rsp+108h] [rbp-118h]
  __int64 v67; // [rsp+110h] [rbp-110h]
  __int64 v68; // [rsp+118h] [rbp-108h]
  __int64 v69; // [rsp+120h] [rbp-100h]
  char v70; // [rsp+128h] [rbp-F8h]
  __m128i v71; // [rsp+130h] [rbp-F0h] BYREF
  __m128i v72; // [rsp+140h] [rbp-E0h] BYREF
  unsigned __int64 *v73; // [rsp+150h] [rbp-D0h] BYREF
  char v74; // [rsp+158h] [rbp-C8h]
  int v75; // [rsp+1B0h] [rbp-70h]
  __int64 v76; // [rsp+1B8h] [rbp-68h]
  __m128i v77; // [rsp+1D0h] [rbp-50h] BYREF
  __m128i v78[4]; // [rsp+1E0h] [rbp-40h] BYREF

  v2 = *(_QWORD *)(a1 + 384);
  v55 = 0;
  v56 = 0;
  v57 = 0;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v63 = 0;
  v64 = 0;
  v65 = 0;
  v66 = 0;
  v67 = 0;
  v68 = v2;
  v69 = v2;
  v70 = a2;
  v71 = 0u;
  v72 = 0u;
  v49 = 0;
  if ( (*(_BYTE *)a1 & 0x10) == 0 )
  {
    if ( (unsigned __int8)sub_2BE0770(a1) )
    {
      v37 = *(_BYTE **)(a1 + 272);
      LOBYTE(v49) = 1;
      HIBYTE(v49) = *v37;
    }
    else if ( *(_DWORD *)(a1 + 152) == 28 && (unsigned __int8)sub_2BE0030(a1) )
    {
      v49 = 11521;
    }
  }
  while ( (unsigned __int8)sub_2BEB210(a1, (unsigned __int8 *)&v49, (__int64)&v55) )
    ;
  if ( (_BYTE)v49 )
  {
    LOBYTE(v73) = HIBYTE(v49);
    sub_2BE78D0((__int64)&v55, (char *)&v73);
  }
  v3 = (char *)v55;
  v4 = v56;
  v5 = (char *)v55;
  if ( v56 != (char *)v55 )
  {
    _BitScanReverse64(&v6, (unsigned __int64)&v56[-v55]);
    sub_2BDC070(v55, v56, 2LL * (int)(63 - (v6 ^ 0x3F)));
    sub_2BDBF50(v3, v4);
    v3 = v56;
    v5 = (char *)v55;
  }
  v7 = 0;
  v8 = sub_2BDBC30(v5, v3);
  sub_2BE3780((__int64)&v55, v8, v56);
  do
  {
    v73 = &v55;
    v74 = v7;
    v10 = sub_2BE41C0((char *)&v73);
    v11 = v70;
    v12 = v10;
    v13 = v7 >> 6;
    v14 = 1LL << v7;
    if ( v12 == v70 )
      v9 = v71.m128i_i64[v13] & ~v14;
    else
      v9 = v71.m128i_i64[v13] | v14;
    ++v7;
    v71.m128i_i64[v13] = v9;
  }
  while ( v7 != 256 );
  v15 = v56;
  v56 = 0;
  v16 = v57;
  v57 = 0;
  v39 = v15;
  v40 = v16;
  v46 = v55;
  v38 = *(unsigned __int64 **)(a1 + 256);
  v48 = (unsigned __int64)v61;
  v41 = v60;
  v17 = (unsigned __int64)v58;
  v42 = v63;
  v18 = v59;
  v43 = v65;
  v19 = v62;
  v45 = v64;
  v55 = 0;
  v60 = 0;
  v59 = 0;
  v58 = 0;
  v63 = 0;
  v62 = 0;
  v61 = 0;
  v44 = v66;
  v20 = _mm_loadu_si128(&v71);
  v21 = _mm_loadu_si128(&v72);
  v66 = 0;
  v75 = v67;
  v22 = v69;
  v77 = v20;
  v65 = 0;
  v64 = 0;
  v76 = v68;
  v53 = 0;
  v78[0] = v21;
  v23 = sub_22077B0(0xA0u);
  if ( v23 )
  {
    v24 = _mm_loadu_si128(&v77);
    *(_QWORD *)(v23 + 16) = v40;
    v25 = v75;
    v26 = _mm_loadu_si128(v78);
    *(_QWORD *)(v23 + 64) = v42;
    *(_QWORD *)v23 = v46;
    *(_DWORD *)(v23 + 96) = v25;
    v27 = v76;
    *(_QWORD *)(v23 + 8) = v39;
    *(_QWORD *)(v23 + 40) = v41;
    *(_QWORD *)(v23 + 48) = v48;
    *(_QWORD *)(v23 + 72) = v45;
    *(_QWORD *)(v23 + 80) = v43;
    *(_QWORD *)(v23 + 88) = v44;
    *(_QWORD *)(v23 + 104) = v27;
    *(_BYTE *)(v23 + 120) = v11;
    v45 = 0;
    v48 = 0;
    v46 = 0;
    *(_QWORD *)(v23 + 24) = v17;
    v17 = 0;
    *(_QWORD *)(v23 + 32) = v18;
    v18 = 0;
    *(_QWORD *)(v23 + 56) = v19;
    v19 = 0;
    *(_QWORD *)(v23 + 112) = v22;
    *(__m128i *)(v23 + 128) = v24;
    *(__m128i *)(v23 + 144) = v26;
  }
  v52.m128i_i64[0] = v23;
  v54 = sub_2BDB750;
  v53 = sub_2BDDB90;
  v50.m128i_i64[1] = sub_2BE0EB0(v38, &v52);
  v28 = *(_QWORD *)(a1 + 256);
  v51 = v50.m128i_i64[1];
  v50.m128i_i64[0] = v28;
  sub_2BE3490((unsigned __int64 *)(a1 + 304), &v50);
  if ( v53 )
    v53((unsigned __int64 **)&v52, (const __m128i **)&v52, 3);
  if ( v45 )
    j_j___libc_free_0(v45);
  for ( i = (unsigned __int64 *)v48; i != v19; i += 8 )
  {
    v30 = i[4];
    if ( (unsigned __int64 *)v30 != i + 6 )
      j_j___libc_free_0(v30);
    if ( (unsigned __int64 *)*i != i + 2 )
      j_j___libc_free_0(*i);
  }
  if ( v48 )
    j_j___libc_free_0(v48);
  for ( j = (unsigned __int64 *)v17; j != v18; j += 4 )
  {
    if ( (unsigned __int64 *)*j != j + 2 )
      j_j___libc_free_0(*j);
  }
  if ( v17 )
    j_j___libc_free_0(v17);
  if ( v46 )
    j_j___libc_free_0(v46);
  if ( v64 )
    j_j___libc_free_0(v64);
  v32 = v62;
  v33 = v61;
  if ( v62 != v61 )
  {
    do
    {
      v34 = v33[4];
      if ( (unsigned __int64 *)v34 != v33 + 6 )
        j_j___libc_free_0(v34);
      if ( (unsigned __int64 *)*v33 != v33 + 2 )
        j_j___libc_free_0(*v33);
      v33 += 8;
    }
    while ( v32 != v33 );
    v33 = v61;
  }
  if ( v33 )
    j_j___libc_free_0((unsigned __int64)v33);
  v35 = v59;
  v36 = v58;
  if ( v59 != v58 )
  {
    do
    {
      if ( (unsigned __int64 *)*v36 != v36 + 2 )
        j_j___libc_free_0(*v36);
      v36 += 4;
    }
    while ( v35 != v36 );
    v36 = v58;
  }
  if ( v36 )
    j_j___libc_free_0((unsigned __int64)v36);
  if ( v55 )
    j_j___libc_free_0(v55);
}
