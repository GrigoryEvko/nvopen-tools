// Function: sub_22AA8A0
// Address: 0x22aa8a0
//
void __fastcall sub_22AA8A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rbx
  __int64 v4; // r11
  __int64 v5; // r12
  __int64 v6; // rbx
  __m128i *v7; // r12
  __int64 v8; // rdx
  __m128i *v9; // r13
  __int64 v10; // r8
  __int64 v11; // r9
  __m128i *v12; // r12
  __m256i *v13; // r10
  __m128i *v14; // r11
  __int64 v15; // r14
  unsigned int v16; // r15d
  unsigned int v17; // edi
  __int64 v18; // rax
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rcx
  unsigned __int64 v22; // rax
  __int64 v23; // rsi
  unsigned __int64 v24; // rdx
  const __m128i *v25; // r13
  __m128i *v26; // rax
  int v27; // r13d
  unsigned int v28; // esi
  int v29; // esi
  int v30; // esi
  unsigned int v31; // ecx
  int v32; // eax
  __int64 v33; // rdx
  __int64 v34; // r15
  __int64 v35; // r13
  __int64 v36; // rbx
  __int64 *v37; // r12
  unsigned int v38; // edi
  unsigned int v39; // eax
  unsigned int v40; // esi
  int v41; // r9d
  const void *v42; // rsi
  __int8 *v43; // r13
  __int64 *v44; // rax
  __int64 v45; // rbx
  const __m128i *v46; // r10
  __int64 v47; // r11
  __int64 v48; // r9
  __int64 v49; // r13
  __int64 v50; // rax
  __int64 v51; // rdx
  _QWORD *v52; // r15
  __int64 v53; // rdx
  _QWORD *v54; // r14
  __int64 v55; // rax
  _QWORD *v56; // r12
  int v57; // edx
  __int8 *v58; // rax
  __m128i v59; // xmm5
  int v60; // eax
  int v61; // ecx
  int v62; // ecx
  __int64 v63; // rdi
  unsigned int v64; // r15d
  __int64 v65; // rsi
  __int64 v66; // rax
  __m128i *v67; // rcx
  __int64 v68; // rdi
  __m128i *v69; // rax
  __m128i v70; // xmm7
  __m128i *v71; // rax
  __m128i *v72; // rdi
  __m128i *v73; // rdx
  __int32 v74; // r12d
  int v75; // r15d
  __int64 v76; // [rsp+8h] [rbp-108h]
  __int64 v77; // [rsp+10h] [rbp-100h]
  __int64 v78; // [rsp+10h] [rbp-100h]
  __int64 v79; // [rsp+18h] [rbp-F8h]
  const __m128i *v80; // [rsp+18h] [rbp-F8h]
  __int64 v81; // [rsp+20h] [rbp-F0h]
  const __m128i *v82; // [rsp+20h] [rbp-F0h]
  __m128i *v83; // [rsp+20h] [rbp-F0h]
  __m256i *v85; // [rsp+30h] [rbp-E0h]
  __m128i *v86; // [rsp+30h] [rbp-E0h]
  __m256i *v87; // [rsp+30h] [rbp-E0h]
  __m256i *v88; // [rsp+30h] [rbp-E0h]
  __m128i *v90; // [rsp+40h] [rbp-D0h]
  int v91; // [rsp+40h] [rbp-D0h]
  __m256i *v92; // [rsp+40h] [rbp-D0h]
  __m128i *v93; // [rsp+40h] [rbp-D0h]
  int v94; // [rsp+40h] [rbp-D0h]
  __m128i *v95; // [rsp+40h] [rbp-D0h]
  __int64 v96; // [rsp+48h] [rbp-C8h]
  int v97; // [rsp+48h] [rbp-C8h]
  __m128i v98; // [rsp+50h] [rbp-C0h] BYREF
  __m256i v99; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v100; // [rsp+80h] [rbp-90h]
  __m128i *v101; // [rsp+90h] [rbp-80h] BYREF
  __int64 v102; // [rsp+98h] [rbp-78h]
  __m128i v103[7]; // [rsp+A0h] [rbp-70h] BYREF

  v3 = a1;
  v101 = v103;
  v4 = *(_QWORD *)(a2 + 32);
  v102 = 0x100000000LL;
  if ( v4 != a2 + 24 )
  {
    v5 = v4;
    while ( 1 )
    {
      v6 = v5 - 56;
      if ( !v5 )
        v6 = 0;
      if ( sub_B2FC80(v6) && *(_DWORD *)(v6 + 36) == 3860 )
      {
        v81 = **(_QWORD **)(*(_QWORD *)(v6 + 24) + 16LL);
        v44 = sub_22A9FC0(a3, v81);
        v45 = *(_QWORD *)(v6 + 16);
        v46 = (const __m128i *)v44;
        if ( v45 )
          break;
      }
LABEL_7:
      v5 = *(_QWORD *)(v5 + 8);
      if ( a2 + 24 == v5 )
      {
        v7 = v101;
        v3 = a1;
        v8 = 56LL * (unsigned int)v102;
        v9 = (__m128i *)((char *)v101 + v8);
        sub_22A9C60(v98.m128i_i64, v101, 0x6DB6DB6DB6DB6DB7LL * (v8 >> 3));
        goto LABEL_9;
      }
    }
    v47 = v81;
    v48 = v5;
    while ( 1 )
    {
      while ( 1 )
      {
        v49 = *(_QWORD *)(v45 + 24);
        if ( *(_BYTE *)v49 == 85 )
          break;
LABEL_54:
        v45 = *(_QWORD *)(v45 + 8);
        if ( !v45 )
          goto LABEL_66;
      }
      v50 = *(_DWORD *)(v49 + 4) & 0x7FFFFFF;
      v51 = *(_QWORD *)(v49 - 32 * v50);
      v52 = *(_QWORD **)(v51 + 24);
      if ( *(_DWORD *)(v51 + 32) > 0x40u )
        v52 = (_QWORD *)*v52;
      v53 = *(_QWORD *)(v49 + 32 * (1 - v50));
      v54 = *(_QWORD **)(v53 + 24);
      if ( *(_DWORD *)(v53 + 32) > 0x40u )
        v54 = (_QWORD *)*v54;
      v55 = *(_QWORD *)(v49 + 32 * (2 - v50));
      v56 = *(_QWORD **)(v55 + 24);
      if ( *(_DWORD *)(v55 + 32) > 0x40u )
        v56 = (_QWORD *)*v56;
      v57 = v102;
      if ( HIDWORD(v102) <= (unsigned int)v102 )
      {
        v77 = v48;
        v79 = v47;
        v82 = v46;
        v66 = sub_C8D7D0((__int64)&v101, (__int64)v103, 0, 0x38u, (unsigned __int64 *)&v98, v48);
        v46 = v82;
        v47 = v79;
        v67 = (__m128i *)v66;
        v48 = v77;
        v68 = 56LL * (unsigned int)v102;
        v69 = (__m128i *)(v68 + v66);
        if ( v69 )
        {
          v70 = _mm_loadu_si128(v82);
          v69[1].m128i_i32[0] = 0;
          v69[1].m128i_i32[1] = (int)v52;
          v69[1].m128i_i32[2] = (int)v54;
          v69[1].m128i_i32[3] = (int)v56;
          v69[2].m128i_i64[0] = v79;
          v69[2].m128i_i64[1] = 0;
          v69[3].m128i_i64[0] = v49;
          *v69 = v70;
          v68 = 56LL * (unsigned int)v102;
        }
        v71 = v101;
        v72 = (__m128i *)((char *)v101 + v68);
        if ( v101 != v72 )
        {
          v73 = v67;
          do
          {
            if ( v73 )
            {
              *v73 = _mm_loadu_si128(v71);
              v73[1] = _mm_loadu_si128(v71 + 1);
              v73[2] = _mm_loadu_si128(v71 + 2);
              v73[3].m128i_i64[0] = v71[3].m128i_i64[0];
            }
            v71 = (__m128i *)((char *)v71 + 56);
            v73 = (__m128i *)((char *)v73 + 56);
          }
          while ( v72 != v71 );
          v72 = v101;
        }
        v74 = v98.m128i_i32[0];
        if ( v72 != v103 )
        {
          v76 = v77;
          v78 = v79;
          v80 = v82;
          v83 = v67;
          _libc_free((unsigned __int64)v72);
          v48 = v76;
          v47 = v78;
          v46 = v80;
          v67 = v83;
        }
        LODWORD(v102) = v102 + 1;
        v101 = v67;
        HIDWORD(v102) = v74;
        goto LABEL_54;
      }
      v58 = &v101->m128i_i8[56 * (unsigned int)v102];
      if ( v58 )
      {
        v59 = _mm_loadu_si128(v46);
        *((_DWORD *)v58 + 4) = 0;
        *((_DWORD *)v58 + 5) = (_DWORD)v52;
        *((_DWORD *)v58 + 6) = (_DWORD)v54;
        *((_DWORD *)v58 + 7) = (_DWORD)v56;
        *((_QWORD *)v58 + 4) = v47;
        *((_QWORD *)v58 + 5) = 0;
        *((_QWORD *)v58 + 6) = v49;
        *(__m128i *)v58 = v59;
        v57 = v102;
      }
      LODWORD(v102) = v57 + 1;
      v45 = *(_QWORD *)(v45 + 8);
      if ( !v45 )
      {
LABEL_66:
        v5 = v48;
        goto LABEL_7;
      }
    }
  }
  v7 = v103;
  v9 = v103;
  sub_22A9C60(v98.m128i_i64, v103, 0);
LABEL_9:
  if ( v99.m256i_i64[0] )
    sub_22A83E0((__int64)v7, (__int64)v9, v99.m256i_i64[0], v98.m128i_i64[1]);
  else
    sub_22A86F0((__int64)v7, (__int64)v9);
  j_j___libc_free_0(v99.m256i_u64[0]);
  v12 = v101;
  v13 = &v99;
  v14 = (__m128i *)((char *)v101 + 56 * (unsigned int)v102);
  v96 = v3 + 48;
  if ( v14 != v101 )
  {
    while ( 1 )
    {
      v20 = *(unsigned int *)(v3 + 8);
      v98 = _mm_loadu_si128(v12);
      v21 = 32 * v20;
      *(__m128i *)v99.m256i_i8 = _mm_loadu_si128(v12 + 1);
      *(__m128i *)&v99.m256i_u64[2] = _mm_loadu_si128(v12 + 2);
      v100 = v12[3].m128i_i64[0];
      v22 = *(_QWORD *)v3;
      if ( !(_DWORD)v20 )
        break;
      v23 = v22 + v21 - 32;
      if ( v99.m256i_i64[0] != *(_QWORD *)v23 || *(_OWORD *)&v99.m256i_u64[1] != *(_OWORD *)(v23 + 8) )
        break;
      v27 = v20 - 1;
      if ( v99.m256i_i64[3] != *(_QWORD *)(v23 + 24) )
      {
        v24 = v20 + 1;
        v25 = (const __m128i *)v13;
        if ( v24 > *(unsigned int *)(v3 + 12) )
        {
LABEL_45:
          v42 = (const void *)(v3 + 16);
          if ( v22 > (unsigned __int64)v13 || (unsigned __int64)v13 >= v22 + v21 )
          {
            v87 = v13;
            v93 = v14;
            sub_C8D5F0(v3, v42, v24, 0x20u, v10, v11);
            v22 = *(_QWORD *)v3;
            v13 = v87;
            v14 = v93;
            v21 = 32LL * *(unsigned int *)(v3 + 8);
            v25 = (const __m128i *)v87;
          }
          else
          {
            v43 = &v25->m128i_i8[-v22];
            v86 = v14;
            v92 = v13;
            sub_C8D5F0(v3, v42, v24, 0x20u, v10, v11);
            v22 = *(_QWORD *)v3;
            v13 = v92;
            v14 = v86;
            v25 = (const __m128i *)&v43[*(_QWORD *)v3];
            v21 = 32LL * *(unsigned int *)(v3 + 8);
          }
        }
LABEL_18:
        v26 = (__m128i *)(v21 + v22);
        *v26 = _mm_loadu_si128(v25);
        v26[1] = _mm_loadu_si128(v25 + 1);
        v27 = *(_DWORD *)(v3 + 8);
        *(_DWORD *)(v3 + 8) = v27 + 1;
      }
      v28 = *(_DWORD *)(v3 + 72);
      if ( !v28 )
      {
        ++*(_QWORD *)(v3 + 48);
        goto LABEL_21;
      }
      v15 = v100;
      v11 = v28 - 1;
      v10 = *(_QWORD *)(v3 + 56);
      v16 = ((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4);
      v17 = v11 & v16;
      v18 = v10 + 16LL * ((unsigned int)v11 & v16);
      v19 = *(_QWORD *)v18;
      if ( v100 == *(_QWORD *)v18 )
      {
LABEL_14:
        v12 = (__m128i *)((char *)v12 + 56);
        *(_DWORD *)(v18 + 8) = v27;
        if ( v14 == v12 )
          goto LABEL_26;
      }
      else
      {
        v94 = 1;
        v33 = 0;
        while ( v19 != -4096 )
        {
          if ( !v33 && v19 == -8192 )
            v33 = v18;
          v17 = v11 & (v94 + v17);
          v10 = (unsigned int)(v94 + 1);
          v18 = *(_QWORD *)(v3 + 56) + 16LL * v17;
          v19 = *(_QWORD *)v18;
          if ( v100 == *(_QWORD *)v18 )
            goto LABEL_14;
          ++v94;
        }
        if ( !v33 )
          v33 = v18;
        v60 = *(_DWORD *)(v3 + 64);
        ++*(_QWORD *)(v3 + 48);
        v32 = v60 + 1;
        if ( 4 * v32 < 3 * v28 )
        {
          if ( v28 - *(_DWORD *)(v3 + 68) - v32 <= v28 >> 3 )
          {
            v88 = v13;
            v95 = v14;
            sub_22AA6C0(v96, v28);
            v61 = *(_DWORD *)(v3 + 72);
            if ( !v61 )
            {
LABEL_114:
              ++*(_DWORD *)(v3 + 64);
              BUG();
            }
            v62 = v61 - 1;
            v63 = *(_QWORD *)(v3 + 56);
            v10 = 0;
            v64 = v62 & v16;
            v14 = v95;
            v13 = v88;
            v11 = 1;
            v32 = *(_DWORD *)(v3 + 64) + 1;
            v33 = v63 + 16LL * v64;
            v65 = *(_QWORD *)v33;
            if ( v15 != *(_QWORD *)v33 )
            {
              while ( v65 != -4096 )
              {
                if ( v65 == -8192 && !v10 )
                  v10 = v33;
                v64 = v62 & (v11 + v64);
                v33 = v63 + 16LL * v64;
                v65 = *(_QWORD *)v33;
                if ( v15 == *(_QWORD *)v33 )
                  goto LABEL_23;
                v11 = (unsigned int)(v11 + 1);
              }
              if ( v10 )
                v33 = v10;
            }
          }
          goto LABEL_23;
        }
LABEL_21:
        v85 = v13;
        v90 = v14;
        sub_22AA6C0(v96, 2 * v28);
        v29 = *(_DWORD *)(v3 + 72);
        if ( !v29 )
          goto LABEL_114;
        v30 = v29 - 1;
        v10 = *(_QWORD *)(v3 + 56);
        v14 = v90;
        v13 = v85;
        v31 = v30 & (((unsigned int)v100 >> 9) ^ ((unsigned int)v100 >> 4));
        v32 = *(_DWORD *)(v3 + 64) + 1;
        v33 = v10 + 16LL * v31;
        v15 = *(_QWORD *)v33;
        if ( v100 != *(_QWORD *)v33 )
        {
          v75 = 1;
          v11 = 0;
          while ( v15 != -4096 )
          {
            if ( v15 == -8192 && !v11 )
              v11 = v33;
            v31 = v30 & (v75 + v31);
            v33 = v10 + 16LL * v31;
            v15 = *(_QWORD *)v33;
            if ( v100 == *(_QWORD *)v33 )
              goto LABEL_23;
            ++v75;
          }
          v15 = v100;
          if ( v11 )
            v33 = v11;
        }
LABEL_23:
        *(_DWORD *)(v3 + 64) = v32;
        if ( *(_QWORD *)v33 != -4096 )
          --*(_DWORD *)(v3 + 68);
        v12 = (__m128i *)((char *)v12 + 56);
        *(_QWORD *)v33 = v15;
        *(_DWORD *)(v33 + 8) = 0;
        *(_DWORD *)(v33 + 8) = v27;
        if ( v14 == v12 )
          goto LABEL_26;
      }
    }
    v24 = v20 + 1;
    v25 = (const __m128i *)v13;
    if ( v24 > *(unsigned int *)(v3 + 12) )
      goto LABEL_45;
    goto LABEL_18;
  }
LABEL_26:
  v97 = *(_DWORD *)(v3 + 8);
  *(_DWORD *)(v3 + 88) = v97;
  *(_DWORD *)(v3 + 84) = v97;
  *(_DWORD *)(v3 + 80) = v97;
  if ( v97 )
  {
    v91 = 0;
    v34 = 0;
    v35 = v3;
    do
    {
      v36 = *(_QWORD *)v35 + 32 * v34;
      v37 = sub_22A9FC0(a3, *(_QWORD *)(v36 + 16));
      if ( sub_22A6B80((__int64)v37) && *(_DWORD *)(v35 + 80) == v97 )
      {
        v38 = *(_DWORD *)(v35 + 84);
        v39 = *(_DWORD *)(v35 + 88);
        v40 = v34;
        v41 = 0;
        v91 = 1;
      }
      else if ( sub_22A6B90((__int64)v37) && *(_DWORD *)(v35 + 84) == v97 )
      {
        v39 = *(_DWORD *)(v35 + 88);
        v40 = *(_DWORD *)(v35 + 80);
        v38 = v34;
        v41 = 0;
        v91 = 1;
      }
      else if ( sub_22A6BA0((__int64)v37) )
      {
        v39 = *(_DWORD *)(v35 + 88);
        v38 = *(_DWORD *)(v35 + 84);
        v40 = *(_DWORD *)(v35 + 80);
        if ( v39 == v97 )
        {
          *(_DWORD *)(v35 + 88) = v34;
          v39 = v34;
          v41 = 0;
          v91 = 1;
        }
        else
        {
          v41 = v91++;
        }
      }
      else
      {
        v38 = *(_DWORD *)(v35 + 84);
        v39 = *(_DWORD *)(v35 + 88);
        v40 = *(_DWORD *)(v35 + 80);
        v41 = v91++;
      }
      if ( v39 > v38 )
        v39 = v38;
      *(_DWORD *)(v35 + 84) = v39;
      if ( v39 > v40 )
        v39 = v40;
      ++v34;
      *(_DWORD *)(v35 + 80) = v39;
      *(_DWORD *)v36 = v41;
    }
    while ( v34 != v97 );
  }
  if ( v101 != v103 )
    _libc_free((unsigned __int64)v101);
}
