// Function: sub_CC21A0
// Address: 0xcc21a0
//
void __fastcall sub_CC21A0(__int64 a1, _QWORD *a2, unsigned __int64 a3)
{
  unsigned __int64 v5; // r12
  __int64 v6; // rdx
  char v7; // si
  unsigned __int8 v8; // di
  unsigned __int8 v9; // r8
  __int64 v10; // rcx
  __m128i v11; // xmm2
  __m128i v12; // xmm3
  __m128i v13; // xmm4
  __m128i v14; // xmm5
  __m128i v15; // xmm1
  __m128i v16; // xmm0
  __m128i v17; // xmm6
  __m128i v18; // xmm5
  __m128i v19; // xmm3
  __m128i v20; // xmm4
  __m128i v21; // xmm2
  const __m128i *v22; // r12
  __int64 i; // r14
  __m128i v24; // xmm7
  __m128i v25; // xmm2
  __m128i v26; // xmm3
  __m128i v27; // xmm7
  __m128i v28; // xmm4
  __m128i v29; // xmm5
  __m128i v30; // xmm6
  __m128i v31; // xmm0
  __m128i v32; // xmm1
  __m128i v33; // xmm5
  __m128i v34; // xmm6
  char v35; // dl
  __m128i v36; // xmm7
  __m128i v37; // xmm3
  __m128i v38; // xmm4
  __m128i v39; // xmm2
  __m128i v40; // xmm1
  __m128i v41; // xmm0
  __m128i *v42; // r9
  _QWORD *v43; // r14
  unsigned __int64 v44; // r13
  __int64 v45; // r12
  __int64 v46; // rax
  __m128i v47; // xmm2
  __m128i v48; // xmm3
  __int64 v49; // rdx
  __m128i v50; // xmm6
  __m128i *v51; // r9
  __m128i v52; // xmm1
  _QWORD *v53; // r14
  __m128i v54; // xmm0
  __m128i v55; // xmm5
  unsigned __int64 v56; // r13
  __m128i v57; // xmm6
  __m128i v58; // xmm2
  __int64 v59; // r12
  __m128i v60; // xmm4
  __m128i v61; // xmm3
  __int64 v62; // rax
  unsigned __int64 v63; // r8
  unsigned int v64; // esi
  __int64 v65; // rdi
  unsigned __int64 v66; // r8
  unsigned int v67; // esi
  __int64 v68; // rdi
  __m128i v69; // xmm1
  __m128i v70; // xmm0
  __int64 v71; // rcx
  __m128i v72; // xmm4
  __m128i v73; // xmm5
  __m128i v74; // xmm4
  __m128i v75; // xmm6
  __m128i v76; // xmm5
  __m128i v77; // xmm4
  __m128i v78; // xmm5
  __m128i v79; // xmm3
  __m128i v80; // xmm2
  unsigned __int64 v82; // [rsp-1F8h] [rbp-1F8h]
  __m128i *v83; // [rsp-1D8h] [rbp-1D8h]
  __m128i *v84; // [rsp-1D0h] [rbp-1D0h]
  __m128i v85; // [rsp-1C8h] [rbp-1C8h] BYREF
  __m128i v86; // [rsp-1B8h] [rbp-1B8h] BYREF
  __m128i v87; // [rsp-1A8h] [rbp-1A8h] BYREF
  __m128i v88; // [rsp-198h] [rbp-198h] BYREF
  __m128i v89; // [rsp-188h] [rbp-188h] BYREF
  __m128i v90; // [rsp-178h] [rbp-178h] BYREF
  __m128i v91; // [rsp-168h] [rbp-168h] BYREF
  __m128i v92; // [rsp-158h] [rbp-158h]
  __m128i v93; // [rsp-148h] [rbp-148h]
  __m128i v94; // [rsp-138h] [rbp-138h]
  __m128i v95; // [rsp-128h] [rbp-128h]
  __m128i v96; // [rsp-118h] [rbp-118h] BYREF
  __m128i v97; // [rsp-108h] [rbp-108h] BYREF
  _BYTE v98[80]; // [rsp-F8h] [rbp-F8h] BYREF
  __m128i v99; // [rsp-A8h] [rbp-A8h] BYREF
  __m128i v100; // [rsp-98h] [rbp-98h]
  _BYTE v101[80]; // [rsp-88h] [rbp-88h] BYREF

  if ( !a3 )
  {
    nullsub_2013();
    return;
  }
  v5 = a3;
  v6 = *(unsigned __int8 *)(a1 + 144);
  v7 = *(_BYTE *)(a1 + 138);
  v8 = *(_BYTE *)(a1 + 137);
  v9 = *(_BYTE *)(a1 + 136);
  if ( (_BYTE)v6 )
  {
    if ( v9 + ((unsigned __int64)v8 << 6) )
    {
      v10 = *(_QWORD *)(a1 + 64);
      v11 = _mm_loadu_si128((const __m128i *)(a1 + 72));
      v98[72] = *(_BYTE *)(a1 + 136);
      v12 = _mm_loadu_si128((const __m128i *)(a1 + 88));
      v13 = _mm_loadu_si128((const __m128i *)(a1 + 104));
      *(_QWORD *)v98 = v10;
      v14 = _mm_loadu_si128((const __m128i *)(a1 + 120));
      v15 = _mm_loadu_si128((const __m128i *)(a1 + 32));
      v16 = _mm_loadu_si128((const __m128i *)(a1 + 48));
      *(__m128i *)&v98[8] = v11;
      *(__m128i *)&v98[24] = v12;
      v17 = _mm_loadu_si128((const __m128i *)v98);
      *(__m128i *)&v98[56] = v14;
      v18 = _mm_loadu_si128((const __m128i *)&v98[16]);
      *(__m128i *)&v98[40] = v13;
      v19 = _mm_loadu_si128((const __m128i *)&v98[48]);
      v20 = _mm_loadu_si128((const __m128i *)&v98[32]);
      v98[73] = v7 | (v8 == 0) | 2;
      v21 = _mm_loadu_si128((const __m128i *)&v98[64]);
      v96 = v15;
      v97 = v16;
      v99 = v15;
      v100 = v16;
      *(__m128i *)v101 = v17;
      *(__m128i *)&v101[16] = v18;
      *(__m128i *)&v101[32] = v20;
      *(__m128i *)&v101[48] = v19;
      *(__m128i *)&v101[64] = v21;
      v89 = v15;
      v90 = v16;
      v91 = v17;
      v92 = v18;
      v93 = v20;
      v94 = v19;
      v95 = v21;
    }
    else
    {
      v69 = _mm_loadu_si128((const __m128i *)a1);
      v70 = _mm_loadu_si128((const __m128i *)(a1 + 16));
      v98[72] = 64;
      *(_QWORD *)v98 = 0;
      v6 = (int)v6 - 2;
      v96 = v69;
      v97 = v70;
      v71 = a1 + 32 * v6;
      v99 = v69;
      v72 = _mm_loadu_si128((const __m128i *)(v71 + 145));
      v73 = _mm_loadu_si128((const __m128i *)(v71 + 161));
      v100 = v70;
      v89 = v69;
      *(__m128i *)&v98[8] = v72;
      v74 = _mm_loadu_si128((const __m128i *)(v71 + 177));
      v75 = _mm_loadu_si128((const __m128i *)v98);
      *(__m128i *)&v98[24] = v73;
      v76 = _mm_loadu_si128((const __m128i *)(v71 + 193));
      *(__m128i *)&v98[40] = v74;
      v77 = _mm_loadu_si128((const __m128i *)&v98[32]);
      *(__m128i *)&v98[56] = v76;
      v78 = _mm_loadu_si128((const __m128i *)&v98[16]);
      v79 = _mm_loadu_si128((const __m128i *)&v98[48]);
      v98[73] = v7 | 4;
      v80 = _mm_loadu_si128((const __m128i *)&v98[64]);
      *(__m128i *)v101 = v75;
      *(__m128i *)&v101[16] = v78;
      *(__m128i *)&v101[32] = v77;
      *(__m128i *)&v101[48] = v79;
      *(__m128i *)&v101[64] = v80;
      v90 = v70;
      v91 = v75;
      v92 = v78;
      v93 = v77;
      v94 = v79;
      v95 = v80;
      if ( !v6 )
        goto LABEL_9;
    }
    v82 = v5;
    v22 = (const __m128i *)(a1 + 32 * (v6 - 1) + 145);
    for ( i = v6 - 1; ; --i )
    {
      v24 = _mm_loadu_si128(v22);
      v25 = _mm_loadu_si128(&v89);
      v22 -= 2;
      v26 = _mm_loadu_si128(&v90);
      v85 = v24;
      v27 = _mm_loadu_si128(v22 + 3);
      v99 = v25;
      v86 = v27;
      v100 = v26;
      sub_CC2280(&v99, &v91.m128i_u64[1], v95.m128i_u8[8], v91.m128i_i64[0], v95.m128i_u8[9]);
      v28 = _mm_loadu_si128((const __m128i *)a1);
      v29 = _mm_loadu_si128((const __m128i *)(a1 + 16));
      v30 = _mm_loadu_si128(&v85);
      v87 = v99;
      v31 = _mm_loadu_si128(&v86);
      v96 = v28;
      v97 = v29;
      v32 = _mm_loadu_si128(&v87);
      v33 = _mm_loadu_si128(&v97);
      v88 = v100;
      *(__m128i *)&v98[8] = v30;
      v34 = _mm_loadu_si128(&v96);
      v35 = *(_BYTE *)(a1 + 138);
      *(__m128i *)&v98[24] = v31;
      v36 = _mm_loadu_si128(&v88);
      v37 = _mm_loadu_si128((const __m128i *)&v98[16]);
      *(_QWORD *)v98 = 0;
      v38 = _mm_loadu_si128((const __m128i *)v98);
      *(__m128i *)&v98[40] = v32;
      v98[72] = 64;
      v98[73] = v35 | 4;
      *(__m128i *)&v98[56] = v36;
      v99 = v34;
      v100 = v33;
      *(__m128i *)v101 = v38;
      *(__m128i *)&v101[16] = v37;
      v39 = _mm_loadu_si128((const __m128i *)&v98[32]);
      v40 = _mm_loadu_si128((const __m128i *)&v98[48]);
      v41 = _mm_loadu_si128((const __m128i *)&v98[64]);
      v89 = v34;
      *(__m128i *)&v101[32] = v39;
      *(__m128i *)&v101[48] = v40;
      *(__m128i *)&v101[64] = v41;
      v90 = v33;
      v91 = v38;
      v92 = v37;
      v93 = v39;
      v94 = v40;
      v95 = v41;
      if ( !i )
        break;
    }
    v5 = v82;
LABEL_9:
    v42 = &v99;
    v43 = a2;
    v44 = v5;
    v45 = 0;
    do
    {
      v83 = v42;
      sub_CC22D0(&v89, &v91.m128i_u64[1], v95.m128i_u8[8], v45, v95.m128i_i8[9] | 8u);
      v42 = v83;
      v46 = 64;
      if ( v44 < 0x40 )
        v46 = v44;
      if ( (unsigned int)v46 >= 8 )
      {
        v63 = (unsigned __int64)(v43 + 1) & 0xFFFFFFFFFFFFFFF8LL;
        *v43 = v83->m128i_i64[0];
        *(_QWORD *)((char *)v43 + (unsigned int)v46 - 8) = *(__int64 *)((char *)&v83->m128i_i64[-1] + (unsigned int)v46);
        if ( (((_DWORD)v46 + (_DWORD)v43 - (_DWORD)v63) & 0xFFFFFFF8) >= 8 )
        {
          v64 = 0;
          do
          {
            v65 = v64;
            v64 += 8;
            *(_QWORD *)(v63 + v65) = *(_QWORD *)((char *)v83 - ((char *)v43 - v63) + v65);
          }
          while ( v64 < (((_DWORD)v46 + (_DWORD)v43 - (_DWORD)v63) & 0xFFFFFFF8) );
        }
      }
      else if ( (v46 & 4) != 0 )
      {
        *(_DWORD *)v43 = v83->m128i_i32[0];
        *(_DWORD *)((char *)v43 + (unsigned int)v46 - 4) = *(__int32 *)((char *)&v83->m128i_i32[-1] + (unsigned int)v46);
      }
      else
      {
        *(_BYTE *)v43 = v83->m128i_i8[0];
        if ( (v46 & 2) != 0 )
          *(_WORD *)((char *)v43 + (unsigned int)v46 - 2) = *(__int16 *)((char *)&v83->m128i_i16[-1] + (unsigned int)v46);
      }
      v43 = (_QWORD *)((char *)v43 + v46);
      ++v45;
      v44 -= v46;
    }
    while ( v44 );
    return;
  }
  v47 = _mm_loadu_si128((const __m128i *)(a1 + 104));
  v48 = _mm_loadu_si128((const __m128i *)(a1 + 120));
  v49 = *(_QWORD *)(a1 + 64);
  *(__m128i *)&v101[8] = _mm_loadu_si128((const __m128i *)(a1 + 72));
  v50 = _mm_loadu_si128((const __m128i *)(a1 + 88));
  v51 = &v99;
  v52 = _mm_loadu_si128((const __m128i *)(a1 + 32));
  *(__m128i *)&v101[40] = v47;
  v53 = a2;
  v54 = _mm_loadu_si128((const __m128i *)(a1 + 48));
  *(__m128i *)&v101[24] = v50;
  v55 = _mm_loadu_si128((const __m128i *)&v101[16]);
  v56 = a3;
  *(__m128i *)&v101[56] = v48;
  v57 = _mm_loadu_si128((const __m128i *)&v101[32]);
  v58 = _mm_loadu_si128((const __m128i *)&v101[48]);
  v101[72] = v9;
  v59 = 0;
  *(_QWORD *)v101 = v49;
  v60 = _mm_loadu_si128((const __m128i *)v101);
  v101[73] = v7 | (v8 == 0) | 2;
  v61 = _mm_loadu_si128((const __m128i *)&v101[64]);
  v99 = v52;
  v100 = v54;
  v96 = v52;
  v97 = v54;
  *(__m128i *)v98 = v60;
  *(__m128i *)&v98[16] = v55;
  *(__m128i *)&v98[32] = v57;
  *(__m128i *)&v98[48] = v58;
  *(__m128i *)&v98[64] = v61;
  do
  {
    v84 = v51;
    sub_CC22D0(&v96, &v98[8], v98[72], v59, v98[73] | 8u);
    v51 = v84;
    v62 = 64;
    if ( v56 < 0x40 )
      v62 = v56;
    if ( (unsigned int)v62 >= 8 )
    {
      v66 = (unsigned __int64)(v53 + 1) & 0xFFFFFFFFFFFFFFF8LL;
      *v53 = v84->m128i_i64[0];
      *(_QWORD *)((char *)v53 + (unsigned int)v62 - 8) = *(__int64 *)((char *)&v84->m128i_i64[-1] + (unsigned int)v62);
      if ( (((_DWORD)v62 + (_DWORD)v53 - (_DWORD)v66) & 0xFFFFFFF8) >= 8 )
      {
        v67 = 0;
        do
        {
          v68 = v67;
          v67 += 8;
          *(_QWORD *)(v66 + v68) = *(_QWORD *)((char *)v84 - ((char *)v53 - v66) + v68);
        }
        while ( v67 < (((_DWORD)v62 + (_DWORD)v53 - (_DWORD)v66) & 0xFFFFFFF8) );
      }
    }
    else if ( (v62 & 4) != 0 )
    {
      *(_DWORD *)v53 = v84->m128i_i32[0];
      *(_DWORD *)((char *)v53 + (unsigned int)v62 - 4) = *(__int32 *)((char *)&v84->m128i_i32[-1] + (unsigned int)v62);
    }
    else if ( (_DWORD)v62 )
    {
      *(_BYTE *)v53 = v84->m128i_i8[0];
      if ( (v62 & 2) != 0 )
        *(_WORD *)((char *)v53 + (unsigned int)v62 - 2) = *(__int16 *)((char *)&v84->m128i_i16[-1] + (unsigned int)v62);
    }
    v53 = (_QWORD *)((char *)v53 + v62);
    ++v59;
    v56 -= v62;
  }
  while ( v56 );
}
