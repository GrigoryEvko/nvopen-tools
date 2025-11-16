// Function: sub_332C230
// Address: 0x332c230
//
__int64 __fastcall sub_332C230(__int64 a1, __int64 a2)
{
  __int64 v2; // r14
  __int64 v3; // r13
  __int64 *v4; // rax
  __int64 v5; // rsi
  __m128i v6; // xmm1
  __m128i v7; // xmm2
  __int64 v8; // r15
  __int64 v9; // r9
  __int64 v10; // rbx
  int v11; // eax
  __int64 v12; // rbx
  unsigned int v14; // r13d
  __int64 v15; // r12
  __int64 v16; // rdi
  __int64 v17; // r9
  __int64 v18; // rax
  unsigned int v19; // r9d
  __int64 v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // rax
  __int16 v23; // dx
  __int64 v24; // rax
  __int64 v25; // r9
  __int64 v26; // rdi
  __int64 v27; // rdx
  __int64 v28; // r12
  unsigned int v29; // edx
  bool v30; // bl
  unsigned int v31; // r12d
  __int64 v32; // r15
  unsigned __int64 v33; // rdx
  __int64 v34; // rax
  unsigned __int64 v35; // rax
  unsigned int v36; // edx
  __int64 v37; // rbx
  unsigned __int16 *v38; // rax
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // rcx
  __int64 v42; // r8
  __int64 v43; // r9
  __int64 v44; // rax
  unsigned __int8 v45; // bl
  unsigned __int8 v46; // al
  __int64 v47; // rax
  __int16 v48; // cx
  __int64 v49; // r8
  bool v50; // al
  __int64 v51; // rdi
  __int64 v52; // rsi
  unsigned __int16 *v53; // rcx
  unsigned int v54; // edx
  __int64 v55; // rdi
  unsigned __int64 v56; // rax
  __int64 v57; // rdx
  int v58; // esi
  __int64 v59; // rbx
  __int64 v60; // r14
  __int64 v61; // r9
  unsigned __int64 v62; // rax
  const __m128i *v63; // rbx
  unsigned __int64 v64; // rcx
  unsigned __int64 v65; // rdx
  const __m128i *v66; // rax
  __m128i *v67; // rcx
  int v68; // edi
  __int64 *v69; // rsi
  __int64 v70; // r12
  __int64 v72; // rax
  __int64 v73; // r8
  __int64 v74; // r10
  unsigned int v75; // edx
  __int64 v76; // rcx
  __int64 v77; // rsi
  __int64 v78; // rax
  __int64 v79; // rdx
  __int64 v80; // rdi
  __int64 v81; // rdx
  unsigned __int16 *v82; // rsi
  __int64 v83; // r8
  int v84; // r9d
  int v85; // ecx
  __int64 v86; // rdx
  __int64 v87; // [rsp-10h] [rbp-340h]
  __int64 v88; // [rsp-8h] [rbp-338h]
  __int64 v89; // [rsp+0h] [rbp-330h]
  int v90; // [rsp+8h] [rbp-328h]
  int v91; // [rsp+20h] [rbp-310h]
  __int64 v92; // [rsp+30h] [rbp-300h]
  int v93; // [rsp+30h] [rbp-300h]
  const __m128i *v94; // [rsp+30h] [rbp-300h]
  int v95; // [rsp+30h] [rbp-300h]
  __int64 v96; // [rsp+40h] [rbp-2F0h]
  unsigned int v97; // [rsp+40h] [rbp-2F0h]
  _QWORD *v98; // [rsp+50h] [rbp-2E0h]
  int v99; // [rsp+50h] [rbp-2E0h]
  __int64 v100; // [rsp+58h] [rbp-2D8h]
  unsigned __int64 v101; // [rsp+58h] [rbp-2D8h]
  unsigned int v102; // [rsp+60h] [rbp-2D0h]
  __int64 v103; // [rsp+60h] [rbp-2D0h]
  __int64 v104; // [rsp+60h] [rbp-2D0h]
  __int64 v105; // [rsp+60h] [rbp-2D0h]
  __int64 v106; // [rsp+68h] [rbp-2C8h]
  __int64 v107; // [rsp+68h] [rbp-2C8h]
  unsigned __int64 v108; // [rsp+68h] [rbp-2C8h]
  __int64 v109; // [rsp+A0h] [rbp-290h] BYREF
  int v110; // [rsp+A8h] [rbp-288h]
  unsigned int v111; // [rsp+B0h] [rbp-280h] BYREF
  __int64 v112; // [rsp+B8h] [rbp-278h]
  __int128 v113; // [rsp+C0h] [rbp-270h] BYREF
  __int64 v114; // [rsp+D0h] [rbp-260h]
  __int64 *v115; // [rsp+E0h] [rbp-250h] BYREF
  __int64 v116; // [rsp+E8h] [rbp-248h]
  __int64 v117; // [rsp+F0h] [rbp-240h] BYREF
  __int64 v118; // [rsp+F8h] [rbp-238h]
  unsigned __int64 v119[2]; // [rsp+150h] [rbp-1E0h] BYREF
  _QWORD v120[16]; // [rsp+160h] [rbp-1D0h] BYREF
  __int64 v121; // [rsp+1E0h] [rbp-150h] BYREF
  __int64 *v122; // [rsp+1E8h] [rbp-148h]
  __int64 v123; // [rsp+1F0h] [rbp-140h]
  int v124; // [rsp+1F8h] [rbp-138h]
  char v125; // [rsp+1FCh] [rbp-134h]
  __int64 v126; // [rsp+200h] [rbp-130h] BYREF

  v2 = a2;
  v3 = a1;
  v4 = *(__int64 **)(a2 + 40);
  v5 = *(_QWORD *)(a2 + 80);
  v6 = _mm_loadu_si128((const __m128i *)v4);
  v7 = _mm_loadu_si128((const __m128i *)(v4 + 5));
  v8 = *v4;
  v9 = *((unsigned int *)v4 + 2);
  v109 = v5;
  v10 = v4[5];
  if ( v5 )
  {
    v102 = v9;
    sub_B96E90((__int64)&v109, v5, 1);
    v9 = v102;
  }
  v11 = *(_DWORD *)(v2 + 72);
  v120[0] = v10;
  v12 = *(_QWORD *)(v8 + 56);
  v125 = 1;
  v110 = v11;
  v122 = &v126;
  v119[0] = (unsigned __int64)v120;
  v123 = 0x100000020LL;
  v119[1] = 0x1000000001LL;
  v124 = 0;
  v126 = v2;
  v121 = 1;
  if ( v12 )
  {
    v96 = v2;
    v14 = v9;
    do
    {
      v15 = *(_QWORD *)(v12 + 16);
      if ( *(_DWORD *)(v15 + 24) == 299 && (*(_WORD *)(v15 + 32) & 0x380) == 0 && (*(_BYTE *)(v15 + 33) & 4) == 0 )
      {
        v16 = *(_QWORD *)(v15 + 40);
        if ( v8 == *(_QWORD *)(v16 + 40) && v14 == *(_DWORD *)(v16 + 48) )
        {
          v92 &= 0xFFFFFFFF00000000LL;
          if ( (unsigned __int8)sub_33CFB90(v16, *(_QWORD *)(a1 + 16) + 288LL, v92, 2) )
          {
            if ( !(unsigned __int8)sub_3285B00(v15, (__int64)&v121, (__int64)v119, 0, 0, v17)
              && !(unsigned __int8)sub_33CFFC0(v15, v96) )
            {
              v18 = *(_QWORD *)(v15 + 40);
              v19 = v14;
              v3 = a1;
              v2 = v96;
              v97 = 0;
              v20 = *(_QWORD *)(v18 + 80);
              v21 = *(unsigned int *)(v18 + 88);
              v22 = *(_QWORD *)(v8 + 48) + 16LL * v19;
              v23 = *(_WORD *)v22;
              v24 = *(_QWORD *)(v22 + 8);
              v106 = v21;
              v93 = v20;
              LOWORD(v111) = v23;
              v112 = v24;
              goto LABEL_21;
            }
          }
        }
      }
      v12 = *(_QWORD *)(v12 + 32);
    }
    while ( v12 );
    v9 = v14;
    v3 = a1;
    v2 = v96;
  }
  v25 = *(_QWORD *)(v8 + 48) + 16 * v9;
  v26 = *(_QWORD *)(v3 + 16);
  v27 = *(_QWORD *)(v25 + 8);
  LOWORD(v111) = *(_WORD *)v25;
  v112 = v27;
  v28 = sub_33EDFE0(v26, v111, v27, 1);
  v103 = v28;
  v93 = v28;
  v106 = v29;
  if ( (_WORD)v111 )
    v30 = (unsigned __int16)(v111 - 176) <= 0x34u;
  else
    v30 = sub_3007100((__int64)&v111);
  v31 = *(_DWORD *)(v28 + 96);
  v98 = *(_QWORD **)(*(_QWORD *)(v3 + 16) + 40LL);
  v32 = v98[6];
  sub_2EAC300((__int64)&v113, (__int64)v98, v31, 0);
  if ( v30 )
  {
    LODWORD(v33) = -1;
    v34 = 40LL * (*(_DWORD *)(v32 + 32) + v31);
  }
  else
  {
    v34 = 40LL * (*(_DWORD *)(v32 + 32) + v31);
    v33 = *(_QWORD *)(*(_QWORD *)(v32 + 8) + v34 + 8);
    if ( v33 > 0x3FFFFFFFFFFFFFFBLL )
      LODWORD(v33) = -2;
  }
  v115 = 0;
  v116 = 0;
  v117 = 0;
  v118 = 0;
  v35 = sub_2E7BD70(
          v98,
          2u,
          v33,
          *(unsigned __int8 *)(*(_QWORD *)(v32 + 8) + v34 + 16),
          (int)&v115,
          0,
          v113,
          v114,
          1u,
          0,
          0);
  v15 = sub_33F3F90(
          *(_QWORD *)(v3 + 16),
          (unsigned int)*(_QWORD *)(v3 + 16) + 288,
          0,
          (unsigned int)&v109,
          v6.m128i_i32[0],
          v6.m128i_i32[2],
          v103,
          v106,
          v35);
  v97 = v36;
  v100 = v36;
LABEL_21:
  v37 = sub_2E79000(*(__int64 **)(*(_QWORD *)(v3 + 16) + 40LL));
  v38 = *(unsigned __int16 **)(v2 + 48);
  v39 = *v38;
  v116 = *((_QWORD *)v38 + 1);
  v40 = *(_QWORD *)(v3 + 16);
  LOWORD(v115) = v39;
  v44 = sub_3007410((__int64)&v115, *(__int64 **)(v40 + 64), v39, v41, v42, v43);
  v45 = sub_AE5260(v37, v44);
  v46 = sub_2EAC4F0(*(_QWORD *)(v15 + 112));
  if ( v45 > v46 )
    v45 = v46;
  v47 = *(_QWORD *)(v2 + 48);
  v48 = *(_WORD *)v47;
  v49 = *(_QWORD *)(v47 + 8);
  LOWORD(v115) = v48;
  v116 = v49;
  if ( v48 )
  {
    v50 = (unsigned __int16)(v48 - 17) <= 0xD3u;
  }
  else
  {
    v90 = v49;
    v50 = sub_30070B0((__int64)&v115);
    LODWORD(v49) = v90;
    v48 = 0;
  }
  v51 = *(_QWORD *)(v3 + 8);
  v52 = *(_QWORD *)(v3 + 16);
  if ( v50 )
  {
    v104 = sub_3465D80(v51, v52, v93, v106, v111, v112, v48, v49, *(_OWORD *)&v7);
    v53 = *(unsigned __int16 **)(v2 + 48);
    v115 = 0;
    v116 = 0;
    v55 = *(_QWORD *)(v3 + 16);
    v56 = v54 | v106 & 0xFFFFFFFF00000000LL;
    v117 = 0;
    v118 = 0;
    v57 = *((_QWORD *)v53 + 1);
    v58 = *v53;
    v107 = v56;
    BYTE1(v56) = 1;
    LOBYTE(v56) = v45;
    v59 = v97;
    BYTE4(v114) = 0;
    LODWORD(v114) = 0;
    v101 = v97 | v100 & 0xFFFFFFFF00000000LL;
    v113 = 0u;
    v60 = sub_33F1F00(v55, v58, v57, (unsigned int)&v109, v15, v101, v104, v107, 0, v114, v56, 0, (__int64)&v115, 0);
  }
  else
  {
    v72 = sub_3466750(v51, v52, v93, v106, v111, v112, v7.m128i_i64[0], v7.m128i_i64[1]);
    v115 = 0;
    v105 = v72;
    v74 = *(_QWORD *)(v3 + 16);
    v116 = 0;
    v117 = 0;
    v76 = v45;
    v118 = 0;
    BYTE1(v76) = 1;
    v108 = v75 | v106 & 0xFFFFFFFF00000000LL;
    if ( (_WORD)v111 )
    {
      v77 = 0;
      LOWORD(v78) = word_4456580[(unsigned __int16)v111 - 1];
    }
    else
    {
      v91 = v76;
      v95 = v74;
      v78 = sub_3009970((__int64)&v111, v88, v87, v76, v73);
      LODWORD(v76) = v91;
      LODWORD(v74) = v95;
      v89 = v78;
      v77 = v86;
    }
    v79 = v89;
    v59 = v97;
    LOWORD(v79) = v78;
    BYTE4(v114) = 0;
    v80 = v79;
    v81 = v77;
    v82 = *(unsigned __int16 **)(v2 + 48);
    LODWORD(v114) = 0;
    v83 = *((_QWORD *)v82 + 1);
    v84 = v76;
    v85 = *v82;
    v101 = v97 | v100 & 0xFFFFFFFF00000000LL;
    v113 = 0u;
    v60 = sub_33F1DB0(
            v74,
            1,
            (unsigned int)&v109,
            v85,
            v83,
            v84,
            v15,
            v101,
            v105,
            v108,
            0,
            v114,
            v80,
            v81,
            0,
            (__int64)&v115);
  }
  sub_34161C0(*(_QWORD *)(v3 + 16), v15, v59 | v101 & 0xFFFFFFFF00000000LL, v60, 1);
  v62 = *(unsigned int *)(v60 + 64);
  v63 = *(const __m128i **)(v60 + 40);
  v115 = &v117;
  v64 = 40 * v62;
  v65 = v62;
  v116 = 0x600000000LL;
  v66 = (const __m128i *)((char *)v63 + 40 * v62);
  if ( v64 > 0xF0 )
  {
    v94 = v66;
    v99 = v65;
    sub_C8D5F0((__int64)&v115, &v117, v65, 0x10u, (__int64)&v117, v61);
    v68 = v116;
    v69 = v115;
    LODWORD(v65) = v99;
    v66 = v94;
    v67 = (__m128i *)&v115[2 * (unsigned int)v116];
  }
  else
  {
    v67 = (__m128i *)&v117;
    v68 = 0;
    v69 = &v117;
  }
  if ( v63 != v66 )
  {
    do
    {
      if ( v67 )
        *v67 = _mm_loadu_si128(v63);
      v63 = (const __m128i *)((char *)v63 + 40);
      ++v67;
    }
    while ( v66 != v63 );
    v69 = v115;
    v68 = v116;
  }
  LODWORD(v116) = v68 + v65;
  *v69 = v15;
  *((_DWORD *)v69 + 2) = v97;
  v70 = sub_33EC210(*(_QWORD *)(v3 + 16), v60, v115, (unsigned int)v116);
  if ( v115 != &v117 )
    _libc_free((unsigned __int64)v115);
  if ( (_QWORD *)v119[0] != v120 )
    _libc_free(v119[0]);
  if ( !v125 )
    _libc_free((unsigned __int64)v122);
  if ( v109 )
    sub_B91220((__int64)&v109, v109);
  return v70;
}
