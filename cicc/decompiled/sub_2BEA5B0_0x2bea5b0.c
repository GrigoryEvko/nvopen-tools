// Function: sub_2BEA5B0
// Address: 0x2bea5b0
//
void __fastcall sub_2BEA5B0(__int64 a1, char a2)
{
  _QWORD *v2; // rax
  char *v3; // r12
  char *v4; // r15
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
  char *v15; // r11
  __int64 v16; // rdx
  unsigned __int64 v17; // r14
  unsigned __int64 *v18; // r15
  unsigned __int64 *v19; // r13
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
  unsigned __int64 v30; // rdi
  unsigned __int64 *j; // rbx
  unsigned __int64 *v32; // rbx
  unsigned __int64 *v33; // r12
  unsigned __int64 v34; // rdi
  unsigned __int64 *v35; // rbx
  unsigned __int64 *v36; // r12
  _BYTE *v37; // rax
  unsigned int v38; // r12d
  __int64 v39; // rax
  unsigned __int64 *v40; // [rsp+8h] [rbp-218h]
  char *v41; // [rsp+10h] [rbp-210h]
  __int64 v42; // [rsp+18h] [rbp-208h]
  __int64 v43; // [rsp+20h] [rbp-200h]
  __int64 v44; // [rsp+28h] [rbp-1F8h]
  __int64 v45; // [rsp+30h] [rbp-1F0h]
  __int64 v46; // [rsp+38h] [rbp-1E8h]
  unsigned __int64 v47; // [rsp+40h] [rbp-1E0h]
  unsigned __int64 v48; // [rsp+48h] [rbp-1D8h]
  unsigned __int64 v50; // [rsp+58h] [rbp-1C8h]
  __int16 v51; // [rsp+6Eh] [rbp-1B2h] BYREF
  __m128i v52; // [rsp+70h] [rbp-1B0h] BYREF
  __int64 v53; // [rsp+80h] [rbp-1A0h]
  __m128i v54; // [rsp+90h] [rbp-190h] BYREF
  __int64 (__fastcall *v55)(unsigned __int64 **, const __m128i **, int); // [rsp+A0h] [rbp-180h]
  bool (__fastcall *v56)(_QWORD *, _BYTE *); // [rsp+A8h] [rbp-178h]
  unsigned __int64 v57; // [rsp+B0h] [rbp-170h] BYREF
  char *v58; // [rsp+B8h] [rbp-168h]
  __int64 v59; // [rsp+C0h] [rbp-160h]
  unsigned __int64 *v60; // [rsp+C8h] [rbp-158h]
  unsigned __int64 *v61; // [rsp+D0h] [rbp-150h]
  __int64 v62; // [rsp+D8h] [rbp-148h]
  unsigned __int64 *v63; // [rsp+E0h] [rbp-140h]
  unsigned __int64 *v64; // [rsp+E8h] [rbp-138h]
  __int64 v65; // [rsp+F0h] [rbp-130h]
  unsigned __int64 v66; // [rsp+F8h] [rbp-128h]
  __int64 v67; // [rsp+100h] [rbp-120h]
  __int64 v68; // [rsp+108h] [rbp-118h]
  __int64 v69; // [rsp+110h] [rbp-110h]
  _QWORD *v70; // [rsp+118h] [rbp-108h]
  _QWORD *v71; // [rsp+120h] [rbp-100h]
  char v72; // [rsp+128h] [rbp-F8h]
  __m128i v73; // [rsp+130h] [rbp-F0h] BYREF
  __m128i v74; // [rsp+140h] [rbp-E0h] BYREF
  unsigned __int64 *v75; // [rsp+150h] [rbp-D0h] BYREF
  char v76; // [rsp+158h] [rbp-C8h]
  int v77; // [rsp+1B0h] [rbp-70h]
  _QWORD *v78; // [rsp+1B8h] [rbp-68h]
  __m128i v79; // [rsp+1D0h] [rbp-50h] BYREF
  __m128i v80[4]; // [rsp+1E0h] [rbp-40h] BYREF

  v2 = *(_QWORD **)(a1 + 384);
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
  v68 = 0;
  v69 = 0;
  v70 = v2;
  v71 = v2;
  v72 = a2;
  v73 = 0u;
  v74 = 0u;
  v51 = 0;
  if ( (*(_BYTE *)a1 & 0x10) == 0 )
  {
    if ( (unsigned __int8)sub_2BE0770(a1) )
    {
      v37 = *(_BYTE **)(a1 + 272);
      LOBYTE(v51) = 1;
      HIBYTE(v51) = *v37;
    }
    else if ( *(_DWORD *)(a1 + 152) == 28 && (unsigned __int8)sub_2BE0030(a1) )
    {
      v51 = 11521;
    }
  }
  while ( (unsigned __int8)sub_2BE9F40(a1, &v51, (__int64)&v57) )
    ;
  if ( (_BYTE)v51 )
  {
    v38 = SHIBYTE(v51);
    v39 = sub_222F790(v70, (__int64)&v51);
    LOBYTE(v75) = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v39 + 32LL))(v39, v38);
    sub_2BE78D0((__int64)&v57, (char *)&v75);
  }
  v3 = (char *)v57;
  v4 = v58;
  v5 = (char *)v57;
  if ( v58 != (char *)v57 )
  {
    _BitScanReverse64(&v6, (unsigned __int64)&v58[-v57]);
    sub_2BDC070(v57, v58, 2LL * (int)(63 - (v6 ^ 0x3F)));
    sub_2BDBF50(v3, v4);
    v3 = v58;
    v5 = (char *)v57;
  }
  v7 = 0;
  v8 = sub_2BDBC30(v5, v3);
  sub_2BE3780((__int64)&v57, v8, v58);
  do
  {
    v75 = &v57;
    v76 = v7;
    v10 = sub_2BE5540((char *)&v75, (__int64)v8);
    v11 = v72;
    v12 = v10;
    v13 = v7 >> 6;
    v14 = 1LL << v7;
    if ( v12 == v72 )
      v9 = v73.m128i_i64[v13] & ~v14;
    else
      v9 = v73.m128i_i64[v13] | v14;
    ++v7;
    v73.m128i_i64[v13] = v9;
  }
  while ( v7 != 256 );
  v15 = v58;
  v58 = 0;
  v16 = v59;
  v59 = 0;
  v41 = v15;
  v42 = v16;
  v48 = v57;
  v40 = *(unsigned __int64 **)(a1 + 256);
  v50 = (unsigned __int64)v63;
  v43 = v62;
  v17 = (unsigned __int64)v60;
  v44 = v65;
  v18 = v61;
  v45 = v67;
  v19 = v64;
  v47 = v66;
  v57 = 0;
  v62 = 0;
  v61 = 0;
  v60 = 0;
  v65 = 0;
  v64 = 0;
  v63 = 0;
  v46 = v68;
  v20 = _mm_loadu_si128(&v73);
  v21 = _mm_loadu_si128(&v74);
  v68 = 0;
  v77 = v69;
  v22 = v71;
  v79 = v20;
  v67 = 0;
  v66 = 0;
  v78 = v70;
  v55 = 0;
  v80[0] = v21;
  v23 = sub_22077B0(0xA0u);
  if ( v23 )
  {
    v24 = _mm_loadu_si128(&v79);
    *(_QWORD *)(v23 + 16) = v42;
    v25 = v77;
    v26 = _mm_loadu_si128(v80);
    *(_QWORD *)(v23 + 64) = v44;
    *(_QWORD *)v23 = v48;
    *(_DWORD *)(v23 + 96) = v25;
    v27 = v78;
    *(_QWORD *)(v23 + 8) = v41;
    *(_QWORD *)(v23 + 40) = v43;
    *(_QWORD *)(v23 + 48) = v50;
    *(_QWORD *)(v23 + 72) = v47;
    *(_QWORD *)(v23 + 80) = v45;
    *(_QWORD *)(v23 + 88) = v46;
    *(_QWORD *)(v23 + 104) = v27;
    *(_BYTE *)(v23 + 120) = v11;
    v47 = 0;
    v50 = 0;
    v48 = 0;
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
  v54.m128i_i64[0] = v23;
  v56 = sub_2BDB7B0;
  v55 = sub_2BDC940;
  v52.m128i_i64[1] = sub_2BE0EB0(v40, &v54);
  v28 = *(_QWORD *)(a1 + 256);
  v53 = v52.m128i_i64[1];
  v52.m128i_i64[0] = v28;
  sub_2BE3490((unsigned __int64 *)(a1 + 304), &v52);
  if ( v55 )
    v55((unsigned __int64 **)&v54, (const __m128i **)&v54, 3);
  if ( v47 )
    j_j___libc_free_0(v47);
  for ( i = (unsigned __int64 *)v50; i != v19; i += 8 )
  {
    v30 = i[4];
    if ( (unsigned __int64 *)v30 != i + 6 )
      j_j___libc_free_0(v30);
    if ( (unsigned __int64 *)*i != i + 2 )
      j_j___libc_free_0(*i);
  }
  if ( v50 )
    j_j___libc_free_0(v50);
  for ( j = (unsigned __int64 *)v17; j != v18; j += 4 )
  {
    if ( (unsigned __int64 *)*j != j + 2 )
      j_j___libc_free_0(*j);
  }
  if ( v17 )
    j_j___libc_free_0(v17);
  if ( v48 )
    j_j___libc_free_0(v48);
  if ( v66 )
    j_j___libc_free_0(v66);
  v32 = v64;
  v33 = v63;
  if ( v64 != v63 )
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
    v33 = v63;
  }
  if ( v33 )
    j_j___libc_free_0((unsigned __int64)v33);
  v35 = v61;
  v36 = v60;
  if ( v61 != v60 )
  {
    do
    {
      if ( (unsigned __int64 *)*v36 != v36 + 2 )
        j_j___libc_free_0(*v36);
      v36 += 4;
    }
    while ( v35 != v36 );
    v36 = v60;
  }
  if ( v36 )
    j_j___libc_free_0((unsigned __int64)v36);
  if ( v57 )
    j_j___libc_free_0(v57);
}
