// Function: sub_229EF00
// Address: 0x229ef00
//
void __fastcall sub_229EF00(__int64 a1)
{
  __int64 v2; // rdx
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  unsigned __int64 v16; // rax
  _BYTE *v17; // rsi
  char *v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // r9
  __int64 v21; // rcx
  __int64 v22; // r8
  unsigned __int64 v23; // r14
  __int64 v24; // rax
  unsigned __int64 v25; // rdi
  __m128i *v26; // rdx
  const __m128i *v27; // rax
  const __m128i *v28; // rcx
  unsigned __int64 v29; // r8
  unsigned __int64 v30; // r13
  __int64 v31; // rax
  unsigned __int64 v32; // rdi
  __m128i *v33; // rdx
  const __m128i *v34; // rax
  unsigned __int64 v35; // rax
  unsigned __int64 v36; // rsi
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  __int64 v40; // r9
  unsigned __int64 v41; // rcx
  unsigned __int64 v42; // rax
  char v43; // si
  unsigned __int64 v44[16]; // [rsp+20h] [rbp-320h] BYREF
  __int64 v45; // [rsp+A0h] [rbp-2A0h] BYREF
  _QWORD *v46; // [rsp+A8h] [rbp-298h]
  __int64 v47; // [rsp+B0h] [rbp-290h]
  int v48; // [rsp+B8h] [rbp-288h]
  char v49; // [rsp+BCh] [rbp-284h]
  _QWORD v50[8]; // [rsp+C0h] [rbp-280h] BYREF
  unsigned __int64 v51; // [rsp+100h] [rbp-240h] BYREF
  unsigned __int64 v52; // [rsp+108h] [rbp-238h]
  unsigned __int64 v53; // [rsp+110h] [rbp-230h]
  char v54[8]; // [rsp+120h] [rbp-220h] BYREF
  unsigned __int64 v55; // [rsp+128h] [rbp-218h]
  char v56; // [rsp+13Ch] [rbp-204h]
  _BYTE v57[64]; // [rsp+140h] [rbp-200h] BYREF
  unsigned __int64 v58; // [rsp+180h] [rbp-1C0h]
  unsigned __int64 v59; // [rsp+188h] [rbp-1B8h]
  unsigned __int64 v60; // [rsp+190h] [rbp-1B0h]
  char v61[8]; // [rsp+1A0h] [rbp-1A0h] BYREF
  unsigned __int64 v62; // [rsp+1A8h] [rbp-198h]
  char v63; // [rsp+1BCh] [rbp-184h]
  _BYTE v64[64]; // [rsp+1C0h] [rbp-180h] BYREF
  unsigned __int64 v65; // [rsp+200h] [rbp-140h]
  unsigned __int64 i; // [rsp+208h] [rbp-138h]
  unsigned __int64 v67; // [rsp+210h] [rbp-130h]
  __m128i v68; // [rsp+220h] [rbp-120h] BYREF
  char v69; // [rsp+230h] [rbp-110h]
  char v70; // [rsp+23Ch] [rbp-104h]
  char v71[64]; // [rsp+240h] [rbp-100h] BYREF
  unsigned __int64 v72; // [rsp+280h] [rbp-C0h]
  __int64 v73; // [rsp+288h] [rbp-B8h]
  unsigned __int64 v74; // [rsp+290h] [rbp-B0h]
  char v75[8]; // [rsp+298h] [rbp-A8h] BYREF
  unsigned __int64 v76; // [rsp+2A0h] [rbp-A0h]
  char v77; // [rsp+2B4h] [rbp-8Ch]
  char v78[64]; // [rsp+2B8h] [rbp-88h] BYREF
  const __m128i *v79; // [rsp+2F8h] [rbp-48h]
  const __m128i *v80; // [rsp+300h] [rbp-40h]
  unsigned __int64 v81; // [rsp+308h] [rbp-38h]

  v2 = **(_QWORD **)(a1 + 8);
  v46 = v50;
  memset(v44, 0, 0x78u);
  v44[1] = (unsigned __int64)&v44[4];
  v3 = *(_QWORD *)(v2 + 120);
  v47 = 0x100000008LL;
  LODWORD(v44[2]) = 8;
  v50[0] = v3;
  v68.m128i_i64[0] = v3;
  BYTE4(v44[3]) = 1;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v48 = 0;
  v49 = 1;
  v45 = 1;
  v69 = 0;
  sub_229D530((__int64)&v51, &v68);
  sub_C8CF70((__int64)v61, v64, 8, (__int64)&v44[4], (__int64)v44);
  v4 = v44[12];
  memset(&v44[12], 0, 24);
  v65 = v4;
  i = v44[13];
  v67 = v44[14];
  sub_C8CF70((__int64)v54, v57, 8, (__int64)v50, (__int64)&v45);
  v5 = v51;
  v51 = 0;
  v58 = v5;
  v6 = v52;
  v52 = 0;
  v59 = v6;
  v7 = v53;
  v53 = 0;
  v60 = v7;
  sub_C8CF70((__int64)&v68, v71, 8, (__int64)v57, (__int64)v54);
  v8 = v58;
  v58 = 0;
  v72 = v8;
  v9 = v59;
  v59 = 0;
  v73 = v9;
  v10 = v60;
  v60 = 0;
  v74 = v10;
  sub_C8CF70((__int64)v75, v78, 8, (__int64)v64, (__int64)v61);
  v14 = v65;
  v65 = 0;
  v79 = (const __m128i *)v14;
  v15 = i;
  i = 0;
  v80 = (const __m128i *)v15;
  v16 = v67;
  v67 = 0;
  v81 = v16;
  if ( v58 )
    j_j___libc_free_0(v58);
  if ( !v56 )
    _libc_free(v55);
  if ( v65 )
    j_j___libc_free_0(v65);
  if ( !v63 )
    _libc_free(v62);
  if ( v51 )
    j_j___libc_free_0(v51);
  if ( !v49 )
    _libc_free((unsigned __int64)v46);
  if ( v44[12] )
    j_j___libc_free_0(v44[12]);
  if ( !BYTE4(v44[3]) )
    _libc_free(v44[1]);
  v17 = v57;
  v18 = v54;
  sub_C8CD80((__int64)v54, (__int64)v57, (__int64)&v68, v11, v12, v13);
  v21 = v73;
  v22 = v72;
  v58 = 0;
  v59 = 0;
  v60 = 0;
  v23 = v73 - v72;
  if ( v73 == v72 )
  {
    v25 = 0;
  }
  else
  {
    if ( v23 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_65;
    v24 = sub_22077B0(v73 - v72);
    v21 = v73;
    v22 = v72;
    v25 = v24;
  }
  v58 = v25;
  v59 = v25;
  v60 = v25 + v23;
  if ( v21 != v22 )
  {
    v26 = (__m128i *)v25;
    v27 = (const __m128i *)v22;
    do
    {
      if ( v26 )
      {
        *v26 = _mm_loadu_si128(v27);
        v26[1].m128i_i64[0] = v27[1].m128i_i64[0];
      }
      v27 = (const __m128i *)((char *)v27 + 24);
      v26 = (__m128i *)((char *)v26 + 24);
    }
    while ( v27 != (const __m128i *)v21 );
    v25 += 8 * (((unsigned __int64)&v27[-2].m128i_u64[1] - v22) >> 3) + 24;
  }
  v59 = v25;
  v17 = v64;
  v18 = v61;
  sub_C8CD80((__int64)v61, (__int64)v64, (__int64)v75, v21, v22, v20);
  v28 = v80;
  v29 = (unsigned __int64)v79;
  v65 = 0;
  i = 0;
  v67 = 0;
  v30 = (char *)v80 - (char *)v79;
  if ( v80 == v79 )
  {
    v32 = 0;
    goto LABEL_29;
  }
  if ( v30 > 0x7FFFFFFFFFFFFFF8LL )
LABEL_65:
    sub_4261EA(v18, v17, v19);
  v31 = sub_22077B0((char *)v80 - (char *)v79);
  v28 = v80;
  v29 = (unsigned __int64)v79;
  v32 = v31;
LABEL_29:
  v65 = v32;
  i = v32;
  v67 = v32 + v30;
  if ( (const __m128i *)v29 == v28 )
  {
    v35 = v32;
  }
  else
  {
    v33 = (__m128i *)v32;
    v34 = (const __m128i *)v29;
    do
    {
      if ( v33 )
      {
        *v33 = _mm_loadu_si128(v34);
        v33[1].m128i_i64[0] = v34[1].m128i_i64[0];
      }
      v34 = (const __m128i *)((char *)v34 + 24);
      v33 = (__m128i *)((char *)v33 + 24);
    }
    while ( v28 != v34 );
    v35 = v32 + 8 * (((unsigned __int64)&v28[-2].m128i_u64[1] - v29) >> 3) + 24;
  }
  for ( i = v35; ; v35 = i )
  {
    v41 = v58;
    if ( v59 - v58 != v35 - v32 )
      goto LABEL_36;
    if ( v58 == v59 )
      break;
    v42 = v32;
    while ( *(_QWORD *)v41 == *(_QWORD *)v42 )
    {
      v43 = *(_BYTE *)(v41 + 16);
      if ( v43 != *(_BYTE *)(v42 + 16) || v43 && *(_QWORD *)(v41 + 8) != *(_QWORD *)(v42 + 8) )
        break;
      v41 += 24LL;
      v42 += 24LL;
      if ( v59 == v41 )
        goto LABEL_45;
    }
LABEL_36:
    v36 = *(_QWORD *)(v59 - 24);
    sub_229E1A0((_BYTE *)a1, v36);
    sub_229DC50((__int64)v54, v36, v37, v38, v39, v40);
    v32 = v65;
  }
LABEL_45:
  if ( v32 )
    j_j___libc_free_0(v32);
  if ( !v63 )
    _libc_free(v62);
  if ( v58 )
    j_j___libc_free_0(v58);
  if ( !v56 )
    _libc_free(v55);
  if ( v79 )
    j_j___libc_free_0((unsigned __int64)v79);
  if ( !v77 )
    _libc_free(v76);
  if ( v72 )
    j_j___libc_free_0(v72);
  if ( !v70 )
    _libc_free(v68.m128i_u64[1]);
}
