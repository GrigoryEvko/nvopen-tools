// Function: sub_2E61790
// Address: 0x2e61790
//
void __fastcall sub_2E61790(__int64 a1)
{
  unsigned __int64 v1; // rax
  unsigned __int64 v2; // rax
  __int64 v3; // rax
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rax
  unsigned __int64 v6; // rax
  unsigned __int64 v7; // rax
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r9
  unsigned __int64 v17; // rsi
  const __m128i *v18; // rdi
  unsigned __int64 v19; // r14
  __int64 v20; // rax
  __int64 v21; // rcx
  __m128i *v22; // rdx
  const __m128i *v23; // rax
  __int64 v24; // r8
  __int64 v25; // r9
  const __m128i *v26; // rcx
  unsigned __int64 v27; // r13
  __int64 v28; // rax
  unsigned __int64 v29; // rdi
  __m128i *v30; // rdx
  const __m128i *v31; // rax
  __int64 v32; // r14
  unsigned __int64 v33; // rax
  __int64 v34; // rdx
  int v35; // eax
  __int64 v36; // rcx
  __int64 v37; // r15
  __int64 *v38; // rax
  __int64 *v39; // rdx
  __int64 v40; // r13
  __int64 *v41; // rax
  char v42; // dl
  unsigned __int64 v43; // rdx
  char v44; // cl
  unsigned __int64 v45[16]; // [rsp+10h] [rbp-320h] BYREF
  __m128i v46; // [rsp+90h] [rbp-2A0h] BYREF
  __int64 v47; // [rsp+A0h] [rbp-290h]
  int v48; // [rsp+A8h] [rbp-288h]
  char v49; // [rsp+ACh] [rbp-284h]
  _QWORD v50[8]; // [rsp+B0h] [rbp-280h] BYREF
  unsigned __int64 v51; // [rsp+F0h] [rbp-240h] BYREF
  __int64 v52; // [rsp+F8h] [rbp-238h]
  unsigned __int64 v53; // [rsp+100h] [rbp-230h]
  __int64 v54; // [rsp+110h] [rbp-220h] BYREF
  __int64 *v55; // [rsp+118h] [rbp-218h]
  unsigned int v56; // [rsp+120h] [rbp-210h]
  unsigned int v57; // [rsp+124h] [rbp-20Ch]
  char v58; // [rsp+12Ch] [rbp-204h]
  _BYTE v59[64]; // [rsp+130h] [rbp-200h] BYREF
  unsigned __int64 v60; // [rsp+170h] [rbp-1C0h] BYREF
  __int64 v61; // [rsp+178h] [rbp-1B8h]
  unsigned __int64 v62; // [rsp+180h] [rbp-1B0h]
  char v63[8]; // [rsp+190h] [rbp-1A0h] BYREF
  unsigned __int64 v64; // [rsp+198h] [rbp-198h]
  char v65; // [rsp+1ACh] [rbp-184h]
  _BYTE v66[64]; // [rsp+1B0h] [rbp-180h] BYREF
  unsigned __int64 v67; // [rsp+1F0h] [rbp-140h]
  unsigned __int64 v68; // [rsp+1F8h] [rbp-138h]
  unsigned __int64 v69; // [rsp+200h] [rbp-130h]
  __m128i v70; // [rsp+210h] [rbp-120h] BYREF
  char v71; // [rsp+220h] [rbp-110h]
  char v72; // [rsp+22Ch] [rbp-104h]
  char v73[64]; // [rsp+230h] [rbp-100h] BYREF
  const __m128i *v74; // [rsp+270h] [rbp-C0h]
  unsigned __int64 v75; // [rsp+278h] [rbp-B8h]
  unsigned __int64 v76; // [rsp+280h] [rbp-B0h]
  char v77[8]; // [rsp+288h] [rbp-A8h] BYREF
  unsigned __int64 v78; // [rsp+290h] [rbp-A0h]
  char v79; // [rsp+2A4h] [rbp-8Ch]
  char v80[64]; // [rsp+2A8h] [rbp-88h] BYREF
  const __m128i *v81; // [rsp+2E8h] [rbp-48h]
  const __m128i *v82; // [rsp+2F0h] [rbp-40h]
  unsigned __int64 v83; // [rsp+2F8h] [rbp-38h]

  v46.m128i_i64[1] = (__int64)v50;
  memset(v45, 0, 0x78u);
  v45[1] = (unsigned __int64)&v45[4];
  v47 = 0x100000008LL;
  v50[0] = a1;
  v70.m128i_i64[0] = a1;
  LODWORD(v45[2]) = 8;
  BYTE4(v45[3]) = 1;
  v51 = 0;
  v52 = 0;
  v53 = 0;
  v48 = 0;
  v49 = 1;
  v46.m128i_i64[0] = 1;
  v71 = 0;
  sub_2E61750(&v51, &v70);
  sub_C8CF70((__int64)v63, v66, 8, (__int64)&v45[4], (__int64)v45);
  v1 = v45[12];
  memset(&v45[12], 0, 24);
  v67 = v1;
  v68 = v45[13];
  v69 = v45[14];
  sub_C8CF70((__int64)&v54, v59, 8, (__int64)v50, (__int64)&v46);
  v2 = v51;
  v51 = 0;
  v60 = v2;
  v3 = v52;
  v52 = 0;
  v61 = v3;
  v4 = v53;
  v53 = 0;
  v62 = v4;
  sub_C8CF70((__int64)&v70, v73, 8, (__int64)v59, (__int64)&v54);
  v5 = v60;
  v60 = 0;
  v74 = (const __m128i *)v5;
  v6 = v61;
  v61 = 0;
  v75 = v6;
  v7 = v62;
  v62 = 0;
  v76 = v7;
  sub_C8CF70((__int64)v77, v80, 8, (__int64)v66, (__int64)v63);
  v11 = v67;
  v67 = 0;
  v81 = (const __m128i *)v11;
  v12 = v68;
  v68 = 0;
  v82 = (const __m128i *)v12;
  v13 = v69;
  v69 = 0;
  v83 = v13;
  if ( v60 )
    j_j___libc_free_0(v60);
  if ( !v58 )
    _libc_free((unsigned __int64)v55);
  if ( v67 )
    j_j___libc_free_0(v67);
  if ( !v65 )
    _libc_free(v64);
  if ( v51 )
    j_j___libc_free_0(v51);
  if ( !v49 )
    _libc_free(v46.m128i_u64[1]);
  if ( v45[12] )
    j_j___libc_free_0(v45[12]);
  if ( !BYTE4(v45[3]) )
    _libc_free(v45[1]);
  sub_C8CD80((__int64)&v54, (__int64)v59, (__int64)&v70, v8, v9, v10);
  v17 = v75;
  v18 = v74;
  v60 = 0;
  v61 = 0;
  v62 = 0;
  v19 = v75 - (_QWORD)v74;
  if ( (const __m128i *)v75 == v74 )
  {
    v21 = 0;
  }
  else
  {
    if ( v19 > 0x7FFFFFFFFFFFFFF8LL )
      goto LABEL_80;
    v20 = sub_22077B0(v75 - (_QWORD)v74);
    v17 = v75;
    v18 = v74;
    v21 = v20;
  }
  v60 = v21;
  v61 = v21;
  v62 = v21 + v19;
  if ( (const __m128i *)v17 != v18 )
  {
    v22 = (__m128i *)v21;
    v23 = v18;
    do
    {
      if ( v22 )
      {
        *v22 = _mm_loadu_si128(v23);
        v15 = v23[1].m128i_i64[0];
        v22[1].m128i_i64[0] = v15;
      }
      v23 = (const __m128i *)((char *)v23 + 24);
      v22 = (__m128i *)((char *)v22 + 24);
    }
    while ( v23 != (const __m128i *)v17 );
    v21 += 8 * ((unsigned __int64)((char *)&v23[-2].m128i_u64[1] - (char *)v18) >> 3) + 24;
  }
  v18 = (const __m128i *)v63;
  v61 = v21;
  sub_C8CD80((__int64)v63, (__int64)v66, (__int64)v77, v21, v15, v16);
  v26 = v82;
  v17 = (unsigned __int64)v81;
  v67 = 0;
  v68 = 0;
  v69 = 0;
  v27 = (char *)v82 - (char *)v81;
  if ( v82 != v81 )
  {
    if ( v27 <= 0x7FFFFFFFFFFFFFF8LL )
    {
      v28 = sub_22077B0((char *)v82 - (char *)v81);
      v26 = v82;
      v17 = (unsigned __int64)v81;
      v29 = v28;
      goto LABEL_29;
    }
LABEL_80:
    sub_4261EA(v18, v17, v14);
  }
  v29 = 0;
LABEL_29:
  v67 = v29;
  v30 = (__m128i *)v29;
  v68 = v29;
  v69 = v29 + v27;
  if ( v26 != (const __m128i *)v17 )
  {
    v31 = (const __m128i *)v17;
    do
    {
      if ( v30 )
      {
        *v30 = _mm_loadu_si128(v31);
        v24 = v31[1].m128i_i64[0];
        v30[1].m128i_i64[0] = v24;
      }
      v31 = (const __m128i *)((char *)v31 + 24);
      v30 = (__m128i *)((char *)v30 + 24);
    }
    while ( v26 != v31 );
    v30 = (__m128i *)(v29 + 8 * (((unsigned __int64)&v26[-2].m128i_u64[1] - v17) >> 3) + 24);
  }
  v32 = v61;
  v33 = v60;
  v68 = (unsigned __int64)v30;
  if ( (__m128i *)(v61 - v60) != (__m128i *)((char *)v30 - v29) )
    goto LABEL_36;
LABEL_52:
  if ( v33 != v32 )
  {
    v43 = v29;
    while ( *(_QWORD *)v33 == *(_QWORD *)v43 )
    {
      v44 = *(_BYTE *)(v33 + 16);
      if ( v44 != *(_BYTE *)(v43 + 16) || v44 && *(_QWORD *)(v33 + 8) != *(_QWORD *)(v43 + 8) )
        break;
      v33 += 24LL;
      v43 += 24LL;
      if ( v33 == v32 )
        goto LABEL_59;
    }
LABEL_36:
    v34 = *(_QWORD *)(v32 - 24);
    v35 = 1;
    v36 = *(_QWORD *)v34;
    if ( *(_QWORD *)v34 )
      v35 = *(_DWORD *)(v36 + 168) + 1;
    *(_DWORD *)(v34 + 168) = v35;
    while ( 1 )
    {
      v37 = *(_QWORD *)(v32 - 24);
      if ( *(_BYTE *)(v32 - 8) )
        break;
      v38 = *(__int64 **)(v37 + 32);
      *(_BYTE *)(v32 - 8) = 1;
      *(_QWORD *)(v32 - 16) = v38;
      if ( v38 != *(__int64 **)(v37 + 40) )
        goto LABEL_41;
LABEL_47:
      v61 -= 24;
      v33 = v60;
      v32 = v61;
      if ( v61 == v60 )
        goto LABEL_51;
    }
    while ( 1 )
    {
      v38 = *(__int64 **)(v32 - 16);
      if ( v38 == *(__int64 **)(v37 + 40) )
        goto LABEL_47;
LABEL_41:
      v39 = v38 + 1;
      *(_QWORD *)(v32 - 16) = v38 + 1;
      v40 = *v38;
      if ( !v58 )
        goto LABEL_49;
      v41 = v55;
      v36 = v57;
      v39 = &v55[v57];
      if ( v55 == v39 )
      {
LABEL_75:
        if ( v57 < v56 )
        {
          ++v57;
          *v39 = v40;
          ++v54;
LABEL_50:
          v46.m128i_i64[0] = v40;
          LOBYTE(v47) = 0;
          sub_2E61750(&v60, &v46);
          v33 = v60;
          v32 = v61;
LABEL_51:
          v29 = v67;
          if ( v32 - v33 == v68 - v67 )
            goto LABEL_52;
          goto LABEL_36;
        }
LABEL_49:
        sub_C8CC70((__int64)&v54, v40, (__int64)v39, v36, v24, v25);
        if ( v42 )
          goto LABEL_50;
      }
      else
      {
        while ( v40 != *v41 )
        {
          if ( v39 == ++v41 )
            goto LABEL_75;
        }
      }
    }
  }
LABEL_59:
  if ( v29 )
    j_j___libc_free_0(v29);
  if ( !v65 )
    _libc_free(v64);
  if ( v60 )
    j_j___libc_free_0(v60);
  if ( !v58 )
    _libc_free((unsigned __int64)v55);
  if ( v81 )
    j_j___libc_free_0((unsigned __int64)v81);
  if ( !v79 )
    _libc_free(v78);
  if ( v74 )
    j_j___libc_free_0((unsigned __int64)v74);
  if ( !v72 )
    _libc_free(v70.m128i_u64[1]);
}
