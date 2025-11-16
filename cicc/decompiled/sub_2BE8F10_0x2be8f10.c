// Function: sub_2BE8F10
// Address: 0x2be8f10
//
void __fastcall sub_2BE8F10(__int64 a1, char a2)
{
  _QWORD *v2; // rax
  char *v3; // r12
  char *v4; // r14
  char *v5; // rdi
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rbx
  char *v8; // rsi
  __int64 v9; // rax
  char v10; // al
  char v11; // r12
  char v12; // r8
  unsigned __int64 v13; // rdx
  __int64 v14; // rax
  char *v15; // rdx
  __int64 v16; // r9
  unsigned __int64 v17; // r13
  unsigned __int64 v18; // r14
  unsigned __int64 *v19; // r15
  __m128i v20; // xmm0
  __m128i v21; // xmm1
  _QWORD *v22; // rbx
  __int64 v23; // rax
  __m128i v24; // xmm2
  int v25; // edx
  __m128i v26; // xmm3
  _QWORD *v27; // rdx
  __int64 v28; // rdx
  unsigned __int64 *i; // rbx
  unsigned __int64 *v30; // rbx
  unsigned __int64 *v31; // r12
  _BYTE *v32; // rax
  unsigned int v33; // r12d
  __int64 v34; // rax
  char v35; // al
  char *v36; // rsi
  unsigned __int64 *v37; // [rsp+8h] [rbp-218h]
  char *v38; // [rsp+10h] [rbp-210h]
  __int64 v39; // [rsp+18h] [rbp-208h]
  __int64 v40; // [rsp+20h] [rbp-200h]
  __int64 v41; // [rsp+28h] [rbp-1F8h]
  __int64 v42; // [rsp+30h] [rbp-1F0h]
  __int64 v43; // [rsp+38h] [rbp-1E8h]
  char *v44; // [rsp+40h] [rbp-1E0h]
  unsigned __int64 v45; // [rsp+48h] [rbp-1D8h]
  unsigned __int64 v46; // [rsp+50h] [rbp-1D0h]
  __int16 v48; // [rsp+6Eh] [rbp-1B2h] BYREF
  __m128i v49; // [rsp+70h] [rbp-1B0h] BYREF
  __int64 v50; // [rsp+80h] [rbp-1A0h]
  __m128i v51; // [rsp+90h] [rbp-190h] BYREF
  __int64 (__fastcall *v52)(unsigned __int64 **, const __m128i **, int); // [rsp+A0h] [rbp-180h]
  bool (__fastcall *v53)(_QWORD *, _BYTE *); // [rsp+A8h] [rbp-178h]
  unsigned __int64 v54; // [rsp+B0h] [rbp-170h] BYREF
  char *v55; // [rsp+B8h] [rbp-168h]
  char *v56; // [rsp+C0h] [rbp-160h]
  unsigned __int64 *v57; // [rsp+C8h] [rbp-158h]
  unsigned __int64 *v58; // [rsp+D0h] [rbp-150h]
  __int64 v59; // [rsp+D8h] [rbp-148h]
  unsigned __int64 v60; // [rsp+E0h] [rbp-140h]
  __int64 v61; // [rsp+E8h] [rbp-138h]
  __int64 v62; // [rsp+F0h] [rbp-130h]
  unsigned __int64 v63; // [rsp+F8h] [rbp-128h]
  __int64 v64; // [rsp+100h] [rbp-120h]
  __int64 v65; // [rsp+108h] [rbp-118h]
  __int64 v66; // [rsp+110h] [rbp-110h]
  _QWORD *v67; // [rsp+118h] [rbp-108h]
  _QWORD *v68; // [rsp+120h] [rbp-100h]
  char v69; // [rsp+128h] [rbp-F8h]
  __m128i v70; // [rsp+130h] [rbp-F0h] BYREF
  __m128i v71; // [rsp+140h] [rbp-E0h] BYREF
  unsigned __int64 *v72; // [rsp+150h] [rbp-D0h] BYREF
  char v73; // [rsp+158h] [rbp-C8h]
  int v74; // [rsp+1B0h] [rbp-70h]
  _QWORD *v75; // [rsp+1B8h] [rbp-68h]
  __m128i v76; // [rsp+1D0h] [rbp-50h] BYREF
  __m128i v77[4]; // [rsp+1E0h] [rbp-40h] BYREF

  v2 = *(_QWORD **)(a1 + 384);
  v54 = 0;
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
  v67 = v2;
  v68 = v2;
  v69 = a2;
  v70 = 0u;
  v71 = 0u;
  v48 = 0;
  if ( (*(_BYTE *)a1 & 0x10) == 0 )
  {
    if ( (unsigned __int8)sub_2BE0770(a1) )
    {
      v32 = *(_BYTE **)(a1 + 272);
      LOBYTE(v48) = 1;
      HIBYTE(v48) = *v32;
    }
    else if ( *(_DWORD *)(a1 + 152) == 28 && (unsigned __int8)sub_2BE0030(a1) )
    {
      v48 = 11521;
    }
  }
  while ( (unsigned __int8)sub_2BE8870(a1, &v48, (__int64)&v54) )
    ;
  if ( (_BYTE)v48 )
  {
    v33 = SHIBYTE(v48);
    v34 = sub_222F790(v67, (__int64)&v48);
    v35 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v34 + 32LL))(v34, v33);
    v36 = v55;
    LOBYTE(v72) = v35;
    if ( v55 == v56 )
    {
      sub_17EB120((__int64)&v54, v55, (char *)&v72);
      v3 = v55;
    }
    else
    {
      if ( v55 )
      {
        *v55 = v35;
        v36 = v55;
      }
      v3 = v36 + 1;
      v55 = v36 + 1;
    }
  }
  else
  {
    v3 = v55;
  }
  v4 = (char *)v54;
  v5 = v3;
  if ( (char *)v54 != v3 )
  {
    _BitScanReverse64(&v6, (unsigned __int64)&v3[-v54]);
    sub_2BDC070(v54, v3, 2LL * (int)(63 - (v6 ^ 0x3F)));
    sub_2BDBF50(v4, v3);
    v3 = v55;
    v5 = (char *)v54;
  }
  v7 = 0;
  v8 = sub_2BDBC30(v5, v3);
  sub_2BE3780((__int64)&v54, v8, v55);
  do
  {
    v72 = &v54;
    v73 = v7;
    v10 = sub_2BE4CE0((char *)&v72, (__int64)v8);
    v11 = v69;
    v12 = v10;
    v13 = v7 >> 6;
    v14 = 1LL << v7;
    if ( v12 == v69 )
      v9 = v70.m128i_i64[v13] & ~v14;
    else
      v9 = v70.m128i_i64[v13] | v14;
    ++v7;
    v70.m128i_i64[v13] = v9;
  }
  while ( v7 != 256 );
  v15 = v56;
  v56 = 0;
  v16 = v59;
  v59 = 0;
  v38 = v15;
  v39 = v16;
  v44 = v55;
  v37 = *(unsigned __int64 **)(a1 + 256);
  v45 = v60;
  v40 = v61;
  v17 = v54;
  v41 = v62;
  v18 = (unsigned __int64)v57;
  v42 = v64;
  v19 = v58;
  v46 = v63;
  v55 = 0;
  v54 = 0;
  v58 = 0;
  v57 = 0;
  v62 = 0;
  v61 = 0;
  v60 = 0;
  v43 = v65;
  v20 = _mm_loadu_si128(&v70);
  v21 = _mm_loadu_si128(&v71);
  v65 = 0;
  v74 = v66;
  v22 = v68;
  v76 = v20;
  v64 = 0;
  v63 = 0;
  v75 = v67;
  v52 = 0;
  v77[0] = v21;
  v23 = sub_22077B0(0xA0u);
  if ( v23 )
  {
    v24 = _mm_loadu_si128(&v76);
    *(_QWORD *)(v23 + 16) = v38;
    v25 = v74;
    v26 = _mm_loadu_si128(v77);
    *(_QWORD *)(v23 + 64) = v41;
    *(_QWORD *)(v23 + 8) = v44;
    *(_DWORD *)(v23 + 96) = v25;
    v27 = v75;
    *(_QWORD *)(v23 + 40) = v39;
    *(_QWORD *)(v23 + 48) = v45;
    *(_QWORD *)(v23 + 56) = v40;
    *(_QWORD *)(v23 + 72) = v46;
    *(_QWORD *)(v23 + 80) = v42;
    *(_QWORD *)(v23 + 88) = v43;
    *(_QWORD *)(v23 + 104) = v27;
    *(_BYTE *)(v23 + 120) = v11;
    v46 = 0;
    v45 = 0;
    *(_QWORD *)v23 = v17;
    v17 = 0;
    *(_QWORD *)(v23 + 24) = v18;
    v18 = 0;
    *(_QWORD *)(v23 + 32) = v19;
    v19 = 0;
    *(_QWORD *)(v23 + 112) = v22;
    *(__m128i *)(v23 + 128) = v24;
    *(__m128i *)(v23 + 144) = v26;
  }
  v51.m128i_i64[0] = v23;
  v53 = sub_2BDB780;
  v52 = sub_2BDD6A0;
  v49.m128i_i64[1] = sub_2BE0EB0(v37, &v51);
  v28 = *(_QWORD *)(a1 + 256);
  v50 = v49.m128i_i64[1];
  v49.m128i_i64[0] = v28;
  sub_2BE3490((unsigned __int64 *)(a1 + 304), &v49);
  if ( v52 )
    v52((unsigned __int64 **)&v51, (const __m128i **)&v51, 3);
  if ( v46 )
    j_j___libc_free_0(v46);
  if ( v45 )
    j_j___libc_free_0(v45);
  for ( i = (unsigned __int64 *)v18; i != v19; i += 4 )
  {
    if ( (unsigned __int64 *)*i != i + 2 )
      j_j___libc_free_0(*i);
  }
  if ( v18 )
    j_j___libc_free_0(v18);
  if ( v17 )
    j_j___libc_free_0(v17);
  if ( v63 )
    j_j___libc_free_0(v63);
  if ( v60 )
    j_j___libc_free_0(v60);
  v30 = v58;
  v31 = v57;
  if ( v58 != v57 )
  {
    do
    {
      if ( (unsigned __int64 *)*v31 != v31 + 2 )
        j_j___libc_free_0(*v31);
      v31 += 4;
    }
    while ( v30 != v31 );
    v31 = v57;
  }
  if ( v31 )
    j_j___libc_free_0((unsigned __int64)v31);
  if ( v54 )
    j_j___libc_free_0(v54);
}
