// Function: sub_1FD9F10
// Address: 0x1fd9f10
//
__int64 __fastcall sub_1FD9F10(_QWORD *a1, __int64 a2)
{
  __int64 v2; // r14
  _QWORD *v3; // r12
  char v4; // r15
  unsigned int v5; // r13d
  __int64 v6; // r8
  __int64 v7; // rax
  __int64 v8; // rcx
  unsigned __int8 v9; // al
  int v10; // r9d
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r8
  _QWORD *v14; // rdi
  __int64 (__fastcall *v15)(__int64, unsigned __int8); // rax
  __int64 v16; // rsi
  unsigned __int32 v17; // eax
  __int64 v18; // rdx
  __int64 v19; // rcx
  int v20; // r8d
  int v21; // r9d
  unsigned int v22; // edx
  unsigned int v23; // ecx
  __int64 v24; // rsi
  _QWORD *v25; // rax
  __m128i *v26; // rax
  __int64 v27; // rcx
  __int64 v28; // rax
  __int64 v29; // rdx
  bool v30; // cc
  _QWORD *v31; // rdx
  __m128i *v32; // rax
  __m128i v33; // xmm7
  __int64 v34; // rdx
  unsigned __int8 v35; // al
  __int64 v36; // rdx
  _QWORD *v37; // rax
  int v38; // r8d
  int v39; // r9d
  unsigned int v40; // eax
  __int64 v41; // rax
  __m128i *v42; // rax
  __int64 v43; // rax
  __m128i *v44; // rax
  __m128i v45; // xmm5
  int v46; // r8d
  __int32 *v47; // r9
  __int64 v48; // rax
  _QWORD *v49; // r14
  __int32 *v50; // rbx
  __int32 *v51; // r12
  __int32 v52; // edx
  __m128i *v53; // rax
  __int64 v54; // rdx
  int v55; // r8d
  int v56; // r9d
  __int64 v57; // rdi
  __int64 (*v58)(); // rcx
  __int64 v59; // rax
  __int64 v60; // rdi
  __int64 (*v61)(); // rax
  unsigned __int16 *v62; // rax
  int v63; // r9d
  __int64 v64; // rdx
  __int32 v65; // ecx
  int v66; // r8d
  __int64 v67; // rbx
  unsigned __int16 *v68; // r13
  __m128i *v69; // rdx
  int v70; // esi
  int *v71; // r13
  int *v72; // r8
  __int64 v73; // rax
  int *v74; // rbx
  __int32 v75; // edx
  __m128i *v76; // rax
  __int64 *v77; // rdx
  __int64 v78; // rax
  __int64 v79; // r15
  __int64 v80; // rbx
  _QWORD *v81; // rdx
  _QWORD *v82; // r13
  const __m128i *v83; // r12
  const __m128i *v84; // r15
  const __m128i *v85; // rdx
  unsigned __int64 v86; // rdi
  unsigned int v88; // r15d
  __m128i *v89; // rax
  __m128i v90; // xmm7
  __int32 v91; // eax
  int v92; // r8d
  int v93; // r9d
  __int64 v94; // rax
  __int64 v95; // rax
  __int64 v96; // [rsp-10h] [rbp-890h]
  __int64 v97; // [rsp-8h] [rbp-888h]
  __int64 v98; // [rsp+8h] [rbp-878h]
  unsigned int v99; // [rsp+38h] [rbp-848h]
  __int64 v100; // [rsp+38h] [rbp-848h]
  _QWORD *v101; // [rsp+38h] [rbp-848h]
  __int64 *v102; // [rsp+38h] [rbp-848h]
  unsigned __int8 v103; // [rsp+48h] [rbp-838h]
  __int64 v104; // [rsp+48h] [rbp-838h]
  unsigned int v105; // [rsp+48h] [rbp-838h]
  __int64 v106; // [rsp+48h] [rbp-838h]
  __int64 v107; // [rsp+48h] [rbp-838h]
  __m128i v108; // [rsp+50h] [rbp-830h] BYREF
  __m128i v109; // [rsp+60h] [rbp-820h] BYREF
  __int64 v110; // [rsp+70h] [rbp-810h]
  _QWORD v111[5]; // [rsp+80h] [rbp-800h] BYREF
  __int64 v112; // [rsp+A8h] [rbp-7D8h]
  __int64 v113; // [rsp+B0h] [rbp-7D0h]
  __int64 v114; // [rsp+B8h] [rbp-7C8h]
  __int64 v115; // [rsp+C0h] [rbp-7C0h]
  __int64 *v116; // [rsp+C8h] [rbp-7B8h]
  __int64 v117; // [rsp+D0h] [rbp-7B0h]
  _BYTE *v118; // [rsp+D8h] [rbp-7A8h]
  __int64 v119; // [rsp+E0h] [rbp-7A0h]
  _BYTE v120[128]; // [rsp+E8h] [rbp-798h] BYREF
  _BYTE *v121; // [rsp+168h] [rbp-718h]
  __int64 v122; // [rsp+170h] [rbp-710h]
  _BYTE v123[128]; // [rsp+178h] [rbp-708h] BYREF
  __int32 *v124; // [rsp+1F8h] [rbp-688h]
  __int64 v125; // [rsp+200h] [rbp-680h]
  _BYTE v126[64]; // [rsp+208h] [rbp-678h] BYREF
  _BYTE *v127; // [rsp+248h] [rbp-638h]
  __int64 v128; // [rsp+250h] [rbp-630h]
  _BYTE v129[192]; // [rsp+258h] [rbp-628h] BYREF
  int *v130; // [rsp+318h] [rbp-568h]
  __int64 v131; // [rsp+320h] [rbp-560h]
  _BYTE v132[24]; // [rsp+328h] [rbp-558h] BYREF
  const __m128i *v133; // [rsp+340h] [rbp-540h] BYREF
  __int64 v134; // [rsp+348h] [rbp-538h]
  _BYTE v135[1328]; // [rsp+350h] [rbp-530h] BYREF

  v2 = a2;
  v3 = a1;
  v4 = *(_BYTE *)(*(_QWORD *)a2 + 8LL);
  v5 = (*(unsigned __int16 *)(a2 + 18) >> 2) & 0x3FFFDFFF;
  v6 = sub_1649C60(*(_QWORD *)(a2 + 24 * (2LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF))));
  v7 = *(_QWORD *)(a2 + 24 * (3LL - (*(_DWORD *)(a2 + 20) & 0xFFFFFFF)));
  if ( *(_DWORD *)(v7 + 32) <= 0x40u )
    v8 = *(_QWORD *)(v7 + 24);
  else
    v8 = **(_QWORD **)(v7 + 24);
  v118 = v120;
  v99 = v8;
  if ( v5 == 13 )
    LODWORD(v8) = 0;
  v121 = v123;
  v124 = (__int32 *)v126;
  v119 = 0x1000000000LL;
  v122 = 0x1000000000LL;
  v125 = 0x1000000000LL;
  v127 = v129;
  v111[1] = -4294967200LL;
  v128 = 0x400000000LL;
  v130 = (int *)v132;
  v131 = 0x400000000LL;
  v111[0] = 0;
  memset(&v111[2], 0, 24);
  v112 = 0;
  v113 = 0;
  v114 = 0;
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v98 = v6;
  v9 = sub_1FD72A0(a1, a2, 4u, v8, v6, v5 == 13, (__int64)v111);
  v11 = v96;
  v12 = v97;
  v103 = v9;
  if ( v9 )
  {
    v13 = v98;
    v133 = (const __m128i *)v135;
    v134 = 0x2000000000LL;
    if ( v4 && v5 == 13 )
    {
      v14 = (_QWORD *)a1[14];
      v15 = *(__int64 (__fastcall **)(__int64, unsigned __int8))(*v14 + 288LL);
      if ( v15 == sub_1D45FB0 )
      {
        v16 = v14[21];
      }
      else
      {
        v95 = ((__int64 (__fastcall *)(_QWORD *, __int64, __int64))v15)(v14, 6, v96);
        v13 = v98;
        v16 = v95;
      }
      v104 = v13;
      v17 = sub_1FD3F10((__int64)v3, v16, v11, v12, v13, v10);
      v108.m128i_i64[0] = 0x10000000;
      v117 = v17 | 0x100000000LL;
      v109 = 0u;
      v108.m128i_i32[2] = v17;
      v110 = 0;
      sub_1FD3F30((__int64)&v133, &v108, v18, v19, v20, v21);
      v22 = v134;
      v23 = HIDWORD(v134);
      v13 = v104;
    }
    else
    {
      v23 = 32;
      v22 = 0;
    }
    v24 = *(_QWORD *)(v2 - 24LL * (*(_DWORD *)(v2 + 20) & 0xFFFFFFF));
    v25 = *(_QWORD **)(v24 + 24);
    if ( *(_DWORD *)(v24 + 32) > 0x40u )
      v25 = (_QWORD *)*v25;
    v108.m128i_i64[0] = 1;
    v109.m128i_i64[0] = 0;
    v109.m128i_i64[1] = (__int64)v25;
    if ( v22 >= v23 )
    {
      v107 = v13;
      sub_16CD150((__int64)&v133, v135, 0, 40, v13, v10);
      v22 = v134;
      v13 = v107;
    }
    v26 = (__m128i *)((char *)v133 + 40 * v22);
    *v26 = _mm_loadu_si128(&v108);
    v26[1] = _mm_loadu_si128(&v109);
    v26[2].m128i_i64[0] = v110;
    v27 = *(_DWORD *)(v2 + 20) & 0xFFFFFFF;
    v28 = (unsigned int)(v134 + 1);
    LODWORD(v134) = v134 + 1;
    v29 = *(_QWORD *)(v2 + 24 * (1 - v27));
    v30 = *(_DWORD *)(v29 + 32) <= 0x40u;
    v31 = *(_QWORD **)(v29 + 24);
    if ( !v30 )
      v31 = (_QWORD *)*v31;
    v108.m128i_i64[0] = 1;
    v109.m128i_i64[0] = 0;
    v109.m128i_i64[1] = (__int64)v31;
    if ( (unsigned int)v28 >= HIDWORD(v134) )
    {
      v106 = v13;
      sub_16CD150((__int64)&v133, v135, 0, 40, v13, v10);
      v28 = (unsigned int)v134;
      v13 = v106;
    }
    v32 = (__m128i *)((char *)v133 + 40 * v28);
    *v32 = _mm_loadu_si128(&v108);
    v33 = _mm_loadu_si128(&v109);
    LODWORD(v134) = v134 + 1;
    v32[1] = v33;
    v34 = v110;
    v32[2].m128i_i64[0] = v110;
    v35 = *(_BYTE *)(v13 + 16);
    if ( v35 <= 0x17u )
    {
      if ( v35 != 5 )
      {
        if ( v35 > 3u )
        {
          v108.m128i_i64[0] = 1;
          v109 = 0u;
        }
        else
        {
          v108.m128i_i8[0] = 10;
          v109.m128i_i64[0] = 0;
          v108.m128i_i32[0] &= 0xFFF000FF;
          v109.m128i_i64[1] = v13;
          v108.m128i_i32[2] = 0;
          LODWORD(v110) = 0;
        }
        sub_1FD3F30((__int64)&v133, &v108, v34, v27, v13, v10);
LABEL_23:
        v40 = v99;
        if ( v5 != 13 )
          v40 = v125;
        v108.m128i_i64[0] = 1;
        v109.m128i_i64[0] = 0;
        v109.m128i_i64[1] = v40;
        v41 = (unsigned int)v134;
        if ( (unsigned int)v134 >= HIDWORD(v134) )
        {
          sub_16CD150((__int64)&v133, v135, 0, 40, v38, v39);
          v41 = (unsigned int)v134;
        }
        v42 = (__m128i *)((char *)v133 + 40 * v41);
        *v42 = _mm_loadu_si128(&v108);
        v42[1] = _mm_loadu_si128(&v109);
        v42[2].m128i_i64[0] = v110;
        v108.m128i_i64[0] = 1;
        v43 = (unsigned int)(v134 + 1);
        v109.m128i_i64[1] = v5;
        LODWORD(v134) = v43;
        v109.m128i_i64[0] = 0;
        if ( HIDWORD(v134) <= (unsigned int)v43 )
        {
          sub_16CD150((__int64)&v133, v135, 0, 40, v38, v39);
          v43 = (unsigned int)v134;
        }
        v44 = (__m128i *)((char *)v133 + 40 * v43);
        *v44 = _mm_loadu_si128(&v108);
        v45 = _mm_loadu_si128(&v109);
        LODWORD(v134) = v134 + 1;
        v44[1] = v45;
        v44[2].m128i_i64[0] = v110;
        v105 = v99 + 4;
        if ( v5 == 13 && v99 )
        {
          v88 = 4;
          while ( 1 )
          {
            v91 = sub_1FD8F60(v3, *(_QWORD *)(v2 + 24 * (v88 - (unsigned __int64)(*(_DWORD *)(v2 + 20) & 0xFFFFFFF))));
            if ( !v91 )
              break;
            v108.m128i_i64[0] = 0;
            v108.m128i_i32[2] = v91;
            v94 = (unsigned int)v134;
            v109 = 0u;
            v110 = 0;
            if ( (unsigned int)v134 >= HIDWORD(v134) )
            {
              sub_16CD150((__int64)&v133, v135, 0, 40, v92, v93);
              v94 = (unsigned int)v134;
            }
            ++v88;
            v89 = (__m128i *)((char *)v133 + 40 * v94);
            *v89 = _mm_loadu_si128(&v108);
            v90 = _mm_loadu_si128(&v109);
            LODWORD(v134) = v134 + 1;
            v89[1] = v90;
            v89[2].m128i_i64[0] = v110;
            if ( v88 == v105 )
            {
              v5 = 13;
              goto LABEL_30;
            }
          }
        }
        else
        {
LABEL_30:
          v46 = (int)v124;
          v47 = &v124[(unsigned int)v125];
          if ( v124 != v47 )
          {
            v48 = (unsigned int)v134;
            v100 = v2;
            v49 = v3;
            v50 = v124;
            v51 = &v124[(unsigned int)v125];
            do
            {
              v52 = *v50;
              v108.m128i_i64[0] = 0;
              v109 = 0u;
              v108.m128i_i32[2] = v52;
              v110 = 0;
              if ( HIDWORD(v134) <= (unsigned int)v48 )
              {
                sub_16CD150((__int64)&v133, v135, 0, 40, v46, (int)v47);
                v48 = (unsigned int)v134;
              }
              ++v50;
              v53 = (__m128i *)((char *)v133 + 40 * v48);
              *v53 = _mm_loadu_si128(&v108);
              v53[1] = _mm_loadu_si128(&v109);
              v53[2].m128i_i64[0] = v110;
              v48 = (unsigned int)(v134 + 1);
              LODWORD(v134) = v134 + 1;
            }
            while ( v51 != v50 );
            v3 = v49;
            v2 = v100;
          }
          v103 = sub_1FD9530(v3, (__int64)&v133, v2, v105, v46);
          if ( v103 )
          {
            v57 = v3[15];
            v58 = *(__int64 (**)())(*(_QWORD *)v57 + 32LL);
            v59 = 0;
            if ( v58 != sub_1F49C70 )
              v59 = ((__int64 (__fastcall *)(__int64, _QWORD, _QWORD))v58)(v57, *(_QWORD *)(v3[5] + 8LL), v5);
            v109.m128i_i64[1] = v59;
            v108.m128i_i64[0] = 12;
            v109.m128i_i64[0] = 0;
            sub_1FD3F30((__int64)&v133, &v108, v54, (__int64)v58, v55, v56);
            v60 = v3[14];
            v61 = *(__int64 (**)())(*(_QWORD *)v60 + 1280LL);
            if ( v61 == sub_1FD3440 )
              BUG();
            v62 = (unsigned __int16 *)((__int64 (__fastcall *)(__int64, _QWORD))v61)(v60, v5);
            v64 = (unsigned int)v134;
            v65 = *v62;
            v66 = (int)v62;
            if ( (_WORD)v65 )
            {
              LODWORD(v67) = 0;
              v68 = v62;
              do
              {
                v109 = 0u;
                v108.m128i_i32[2] = v65;
                v110 = 0;
                v108.m128i_i64[0] = 0x430000000LL;
                if ( HIDWORD(v134) <= (unsigned int)v64 )
                {
                  sub_16CD150((__int64)&v133, v135, 0, 40, v66, v63);
                  v64 = (unsigned int)v134;
                }
                v69 = (__m128i *)((char *)v133 + 40 * v64);
                *v69 = _mm_loadu_si128(&v108);
                v70 = v134;
                v69[1] = _mm_loadu_si128(&v109);
                v69[2].m128i_i64[0] = v110;
                v64 = (unsigned int)(v70 + 1);
                LODWORD(v134) = v70 + 1;
                v67 = (unsigned int)(v67 + 1);
                v65 = v68[v67];
              }
              while ( (_WORD)v65 );
            }
            v71 = v130;
            v72 = &v130[(unsigned int)v131];
            if ( v130 != v72 )
            {
              v73 = (unsigned int)v134;
              v74 = &v130[(unsigned int)v131];
              do
              {
                v75 = *v71;
                v108.m128i_i64[0] = 805306368;
                v109 = 0u;
                v108.m128i_i32[2] = v75;
                v110 = 0;
                if ( HIDWORD(v134) <= (unsigned int)v73 )
                {
                  sub_16CD150((__int64)&v133, v135, 0, 40, (int)v72, v63);
                  v73 = (unsigned int)v134;
                }
                ++v71;
                v76 = (__m128i *)((char *)v133 + 40 * v73);
                *v76 = _mm_loadu_si128(&v108);
                v76[1] = _mm_loadu_si128(&v109);
                v76[2].m128i_i64[0] = v110;
                v73 = (unsigned int)(v134 + 1);
                LODWORD(v134) = v134 + 1;
              }
              while ( v74 != v71 );
            }
            v77 = v3 + 10;
            v78 = v3[5];
            v79 = *(_QWORD *)(v78 + 784);
            if ( (*((_BYTE *)v116 + 46) & 4) != 0 )
            {
              v80 = *(_QWORD *)(v79 + 56);
              v102 = v116;
              v82 = sub_1E0B640(v80, *(_QWORD *)(v3[13] + 8LL) + 1344LL, v77, 0);
              sub_1DD6E10(v79, v102, (__int64)v82);
            }
            else
            {
              v80 = sub_1FD3950(*(_QWORD *)(v78 + 784), v116, v77, *(_QWORD *)(v3[13] + 8LL) + 1344LL);
              v82 = v81;
            }
            if ( v133 != (const __m128i *)((char *)v133 + 40 * (unsigned int)v134) )
            {
              v101 = v3;
              v83 = v133;
              v84 = (const __m128i *)((char *)v133 + 40 * (unsigned int)v134);
              do
              {
                v85 = v83;
                v83 = (const __m128i *)((char *)v83 + 40);
                sub_1E1A9C0((__int64)v82, v80, v85);
              }
              while ( v84 != v83 );
              v3 = v101;
            }
            sub_1E1B900((__int64)v82, v130, (unsigned int)v131, v3[15]);
            sub_1E16240((__int64)v116);
            *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(v3[5] + 8LL) + 56LL) + 40LL) = 1;
            if ( HIDWORD(v117) )
            {
              sub_1FD5CC0((__int64)v3, v2, v117, SHIDWORD(v117));
              v86 = (unsigned __int64)v133;
              if ( v133 == (const __m128i *)v135 )
                goto LABEL_58;
LABEL_78:
              _libc_free(v86);
              goto LABEL_58;
            }
LABEL_77:
            v86 = (unsigned __int64)v133;
            if ( v133 == (const __m128i *)v135 )
              goto LABEL_58;
            goto LABEL_78;
          }
        }
        v103 = 0;
        goto LABEL_77;
      }
      v36 = *(_QWORD *)(v13 - 24LL * (*(_DWORD *)(v13 + 20) & 0xFFFFFFF));
      v37 = *(_QWORD **)(v36 + 24);
      if ( *(_DWORD *)(v36 + 32) > 0x40u )
        goto LABEL_21;
    }
    else
    {
      v36 = *(_QWORD *)(v13 - 24);
      v37 = *(_QWORD **)(v36 + 24);
      if ( *(_DWORD *)(v36 + 32) > 0x40u )
LABEL_21:
        v37 = (_QWORD *)*v37;
    }
    v108.m128i_i64[0] = 1;
    v109.m128i_i64[0] = 0;
    v109.m128i_i64[1] = (__int64)v37;
    sub_1FD3F30((__int64)&v133, &v108, v36, v27, v13, v10);
    goto LABEL_23;
  }
LABEL_58:
  if ( v130 != (int *)v132 )
    _libc_free((unsigned __int64)v130);
  if ( v127 != v129 )
    _libc_free((unsigned __int64)v127);
  if ( v124 != (__int32 *)v126 )
    _libc_free((unsigned __int64)v124);
  if ( v121 != v123 )
    _libc_free((unsigned __int64)v121);
  if ( v118 != v120 )
    _libc_free((unsigned __int64)v118);
  if ( v112 )
    j_j___libc_free_0(v112, v114 - v112);
  return v103;
}
