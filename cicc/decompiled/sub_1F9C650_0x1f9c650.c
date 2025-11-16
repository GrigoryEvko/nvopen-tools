// Function: sub_1F9C650
// Address: 0x1f9c650
//
__int64 __fastcall sub_1F9C650(
        __int64 **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __m128 a7,
        double a8,
        __m128i a9)
{
  int v11; // ebx
  __int64 v12; // rax
  __int64 v13; // r14
  void *v14; // rdx
  int v15; // r8d
  int v16; // r9d
  __int64 v17; // r14
  __int64 *v18; // rax
  __int64 v19; // r15
  __int64 v20; // rax
  char v21; // dl
  __int64 v22; // rax
  int v24; // eax
  __int64 v25; // rax
  __int64 v26; // rdx
  char v27; // al
  char v28; // al
  int v29; // r8d
  __int64 *v30; // rax
  __int64 v31; // rbx
  __int64 v32; // r14
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  int v36; // r9d
  __int64 v37; // rdx
  __int64 v38; // rcx
  __int64 v39; // r8
  int v40; // r9d
  __int64 v41; // rdx
  __int64 v42; // rcx
  __int64 v43; // r8
  int v44; // r9d
  __int64 v45; // rdx
  __int64 v46; // rcx
  __int64 v47; // r8
  int v48; // r9d
  __int64 v49; // r15
  __int64 v50; // rdx
  __int64 *v51; // r9
  __int128 *v52; // rbx
  const void **v53; // r8
  __int64 v54; // rcx
  unsigned int v55; // edx
  unsigned int v56; // ebx
  unsigned int v57; // r9d
  unsigned __int16 v58; // bx
  char v59; // al
  unsigned __int8 v60; // si
  __int64 *v61; // rax
  __int64 v62; // rdi
  __int64 v63; // rax
  int v64; // edx
  __int64 v65; // rbx
  int v66; // ecx
  int v67; // r14d
  __int64 v68; // rax
  int v69; // edx
  __int64 v70; // rax
  __int64 v71; // rax
  __int64 v72; // rax
  __int64 v73; // rax
  unsigned __int8 *v74; // r8
  __int64 v75; // rdx
  __int64 v76; // r10
  unsigned int v77; // ecx
  unsigned int v78; // esi
  __int64 v79; // rax
  int v80; // edx
  __int64 v81; // rdx
  __int64 v82; // rcx
  __int64 v83; // r8
  int v84; // r9d
  __int64 v85; // rdx
  __int64 v86; // rcx
  __int64 v87; // r8
  int v88; // r9d
  __int64 v89; // rdx
  __int64 v90; // rcx
  __int64 v91; // r8
  int v92; // r9d
  __int64 v93; // rdx
  __int64 *v94; // r10
  __int64 v95; // rax
  __int64 v96; // rsi
  __int64 v97; // rdi
  const void **v98; // r15
  unsigned int v99; // ebx
  unsigned int v100; // edx
  __int64 v101; // [rsp+10h] [rbp-100h]
  __int64 v102; // [rsp+18h] [rbp-F8h]
  unsigned int v103; // [rsp+20h] [rbp-F0h]
  __int64 v104; // [rsp+28h] [rbp-E8h]
  unsigned int v105; // [rsp+28h] [rbp-E8h]
  __int64 v106; // [rsp+28h] [rbp-E8h]
  __int64 v107; // [rsp+30h] [rbp-E0h]
  unsigned int v108; // [rsp+30h] [rbp-E0h]
  __int64 v109; // [rsp+30h] [rbp-E0h]
  bool v110; // [rsp+3Fh] [rbp-D1h]
  const void **v111; // [rsp+40h] [rbp-D0h]
  __int64 v112; // [rsp+40h] [rbp-D0h]
  int v113; // [rsp+40h] [rbp-D0h]
  __int64 v114; // [rsp+40h] [rbp-D0h]
  __int64 *v115; // [rsp+48h] [rbp-C8h]
  __int64 *v116; // [rsp+48h] [rbp-C8h]
  void *v117; // [rsp+48h] [rbp-C8h]
  __int64 *v118; // [rsp+48h] [rbp-C8h]
  __int64 *v120; // [rsp+50h] [rbp-C0h]
  int v121; // [rsp+50h] [rbp-C0h]
  __int64 v122; // [rsp+58h] [rbp-B8h]
  __int64 v125; // [rsp+90h] [rbp-80h] BYREF
  int v126; // [rsp+98h] [rbp-78h]
  __int128 v127; // [rsp+A0h] [rbp-70h]
  __int64 v128; // [rsp+B0h] [rbp-60h]
  __int64 v129; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v130; // [rsp+C8h] [rbp-48h]
  __int64 v131; // [rsp+D0h] [rbp-40h]

  v11 = a4;
  v12 = sub_1D23470(a3, a4, a3, a4, a5, a6);
  if ( !v12 )
    goto LABEL_14;
  v13 = *(_QWORD *)(v12 + 88);
  v14 = sub_16982C0();
  if ( *(void **)(v13 + 32) == v14 )
    v17 = *(_QWORD *)(v13 + 40) + 8LL;
  else
    v17 = v13 + 32;
  v18 = *(__int64 **)(a2 + 32);
  v19 = *v18;
  if ( (*(_BYTE *)(v17 + 18) & 7) == 1 && *(_WORD *)(a5 + 24) == 164 )
  {
    if ( *(_WORD *)(a2 + 24) == 136 )
    {
      v117 = v14;
      v66 = *(_DWORD *)(v18[20] + 84);
      v113 = v66;
    }
    else
    {
      if ( *(_WORD *)(v19 + 24) != 137 )
        goto LABEL_6;
      v18 = *(__int64 **)(v19 + 32);
      v117 = v14;
      v19 = *v18;
      v66 = *(_DWORD *)(v18[10] + 84);
      v113 = v66;
    }
    v67 = *((_DWORD *)v18 + 2);
    v68 = sub_1D23470(v18[5], v18[6], (__int64)v14, v66, v15, v16);
    v69 = (int)v117;
    if ( v68 )
    {
      v70 = *(_QWORD *)(v68 + 88);
      v71 = v117 == *(void **)(v70 + 32) ? *(_QWORD *)(v70 + 40) + 8LL : v70 + 32;
      if ( (*(_BYTE *)(v71 + 18) & 7) == 3 )
      {
        v72 = *(_QWORD *)(a5 + 32);
        if ( *(_QWORD *)v72 == v19 && *(_DWORD *)(v72 + 8) == v67 )
        {
          LOBYTE(v69) = v113 == 20 || (v113 & 0xFFFFFFF7) == 4;
          LODWORD(v19) = v69;
          if ( (_BYTE)v69 )
          {
            v129 = a5;
            LODWORD(v130) = a6;
            sub_1F994A0((__int64)a1, a2, &v129, 1, 1);
            return (unsigned int)v19;
          }
        }
      }
    }
LABEL_14:
    v18 = *(__int64 **)(a2 + 32);
    v19 = *v18;
  }
LABEL_6:
  v20 = *(_QWORD *)(v19 + 40) + 16LL * *((unsigned int *)v18 + 2);
  v21 = *(_BYTE *)v20;
  v22 = *(_QWORD *)(v20 + 8);
  LOBYTE(v129) = v21;
  v130 = v22;
  if ( v21 )
  {
    LOBYTE(v19) = (unsigned __int8)(v21 - 14) <= 0x5Fu;
  }
  else
  {
    LOBYTE(v24) = sub_1F58D20((__int64)&v129);
    LODWORD(v19) = v24;
  }
  if ( (_BYTE)v19
    || *(_WORD *)(a3 + 24) != *(_WORD *)(a5 + 24)
    || !sub_1D18C00(a3, 1, v11)
    || !sub_1D18C00(a5, 1, a6)
    || *(_WORD *)(a3 + 24) != 185 )
  {
    LODWORD(v19) = 0;
    return (unsigned int)v19;
  }
  v25 = *(_QWORD *)(a5 + 32);
  v26 = *(_QWORD *)(a3 + 32);
  if ( *(_QWORD *)v26 == *(_QWORD *)v25
    && *(_DWORD *)(v26 + 8) == *(_DWORD *)(v25 + 8)
    && (*(_BYTE *)(a3 + 26) & 8) == 0
    && (*(_BYTE *)(a5 + 26) & 8) == 0
    && (*(_WORD *)(a3 + 26) & 0x380) == 0
    && (*(_WORD *)(a5 + 26) & 0x380) == 0 )
  {
    v27 = *(_BYTE *)(a3 + 88);
    if ( *(_BYTE *)(a5 + 88) == v27 && (*(_QWORD *)(a5 + 96) == *(_QWORD *)(a3 + 96) || v27) )
    {
      v28 = (*(_BYTE *)(a3 + 27) >> 2) & 3;
      if ( (v28 == ((*(_BYTE *)(a5 + 27) >> 2) & 3) || ((*(_BYTE *)(a5 + 27) >> 2) & 3) == 1 || v28 == 1)
        && !(unsigned int)sub_1E340A0(*(_QWORD *)(a3 + 104))
        && !(unsigned int)sub_1E340A0(*(_QWORD *)(a5 + 104)) )
      {
        v110 = sub_1F6C880(
                 (__int64)a1[1],
                 *(unsigned __int16 *)(a2 + 24),
                 *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a3 + 32) + 40LL) + 40LL)
                          + 16LL * *(unsigned int *)(*(_QWORD *)(a3 + 32) + 48LL)));
        if ( v110 )
        {
          v30 = *(__int64 **)(a2 + 32);
          v31 = *v30;
          if ( v29 == 134 )
          {
            if ( (unsigned __int8)sub_1D18C40(a3, 1) && (unsigned __int8)sub_1D19270(v31, a3, v81, v82, v83, v84)
              || (unsigned __int8)sub_1D18C40(a5, 1) && (unsigned __int8)sub_1D19270(v31, a5, v85, v86, v87, v88)
              || (unsigned __int8)sub_1D19270(a5, a3, v85, v86, v87, v88)
              || (unsigned __int8)sub_1D19270(a3, a5, v89, v90, v91, v92) )
            {
              return (unsigned int)v19;
            }
            v93 = *(_QWORD *)(a5 + 32);
            v94 = *a1;
            v95 = *(_QWORD *)(a3 + 32);
            v96 = *(_QWORD *)(a2 + 32);
            v97 = *(_QWORD *)(v95 + 40);
            v98 = *(const void ***)(*(_QWORD *)(v97 + 40) + 16LL * *(unsigned int *)(v95 + 48) + 8);
            v99 = *(unsigned __int8 *)(*(_QWORD *)(v97 + 40) + 16LL * *(unsigned int *)(v95 + 48));
            v129 = *(_QWORD *)(a2 + 72);
            if ( v129 )
            {
              v106 = v93;
              v114 = v95;
              v118 = v94;
              sub_1F6CA20(&v129);
              v93 = v106;
              v95 = v114;
              v94 = v118;
            }
            LODWORD(v130) = *(_DWORD *)(a2 + 64);
            v120 = sub_1F810E0(
                     v94,
                     (__int64)&v129,
                     v99,
                     v98,
                     *(_QWORD *)v96,
                     *(__int16 **)(v96 + 8),
                     a7,
                     a8,
                     a9,
                     *(_OWORD *)(v95 + 40),
                     *(_QWORD *)(v93 + 40),
                     *(_QWORD *)(v93 + 48));
            v122 = v100;
            sub_17CD270(&v129);
          }
          else
          {
            v32 = v30[5];
            if ( (unsigned __int8)sub_1D18C40(a3, 1)
              && ((unsigned __int8)sub_1D19270(v31, a3, v33, v34, v35, v36)
               || (unsigned __int8)sub_1D19270(v32, a3, v37, v38, v39, v40))
              || (unsigned __int8)sub_1D18C40(a5, 1)
              && ((unsigned __int8)sub_1D19270(v31, a5, v41, v42, v43, v44)
               || (unsigned __int8)sub_1D19270(v32, a5, v45, v46, v47, v48)) )
            {
              return (unsigned int)v19;
            }
            v49 = *(_QWORD *)(a3 + 32);
            v50 = *(_QWORD *)(a5 + 32);
            v51 = *a1;
            v52 = *(__int128 **)(a2 + 32);
            v53 = *(const void ***)(*(_QWORD *)(*(_QWORD *)(v49 + 40) + 40LL) + 16LL * *(unsigned int *)(v49 + 48) + 8);
            v54 = *(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(v49 + 40) + 40LL) + 16LL * *(unsigned int *)(v49 + 48));
            v129 = *(_QWORD *)(a2 + 72);
            if ( v129 )
            {
              v104 = v54;
              v107 = v50;
              v111 = v53;
              v115 = v51;
              sub_1F6CA20(&v129);
              v54 = v104;
              v50 = v107;
              v53 = v111;
              v51 = v115;
            }
            LODWORD(v130) = *(_DWORD *)(a2 + 64);
            v120 = sub_1D36A20(
                     v51,
                     136,
                     (__int64)&v129,
                     v54,
                     v53,
                     (__int64)v51,
                     *v52,
                     *(__int128 *)((char *)v52 + 40),
                     *(_OWORD *)(v49 + 40),
                     *(_OWORD *)(v50 + 40),
                     v52[10]);
            v122 = v55;
            sub_17CD270(&v129);
          }
          v56 = sub_1E34390(*(_QWORD *)(a5 + 104));
          v57 = sub_1E34390(*(_QWORD *)(a3 + 104));
          if ( v57 >= v56 )
            v57 = v56;
          v58 = *(_WORD *)(*(_QWORD *)(a3 + 104) + 32LL);
          v59 = *(_BYTE *)(a5 + 26);
          if ( (v59 & 0x40) == 0 )
            v58 &= 0x1DFu;
          if ( (v59 & 0x20) == 0 )
            v58 &= 0x1EFu;
          v60 = *(_BYTE *)(a3 + 27);
          v129 = 0;
          v130 = 0;
          v61 = *a1;
          v131 = 0;
          v116 = v61;
          v112 = *(_QWORD *)(a3 + 32);
          v62 = *(_QWORD *)(a2 + 72);
          if ( (v60 & 0xC) != 0 )
          {
            v127 = 0u;
            v73 = *(unsigned __int8 *)(a3 + 88);
            v74 = *(unsigned __int8 **)(a2 + 40);
            v128 = 0;
            v75 = *(_QWORD *)(a3 + 96);
            v76 = *((_QWORD *)v74 + 1);
            v77 = *v74;
            v125 = v62;
            if ( v62 )
            {
              v103 = v77;
              v101 = v73;
              v102 = v75;
              v105 = v57;
              v109 = v76;
              sub_1F6CA20(&v125);
              v60 = *(_BYTE *)(a3 + 27);
              v77 = v103;
              v73 = v101;
              v75 = v102;
              v57 = v105;
              v76 = v109;
            }
            v78 = (v60 >> 2) & 3;
            v126 = *(_DWORD *)(a2 + 64);
            if ( v78 == 1 )
              v78 = (*(_BYTE *)(a5 + 27) >> 2) & 3;
            v79 = sub_1D2B810(
                    v116,
                    v78,
                    (__int64)&v125,
                    v77,
                    v76,
                    v57,
                    *(_OWORD *)v112,
                    (__int64)v120,
                    v122,
                    v127,
                    v128,
                    v73,
                    v75,
                    v58,
                    (__int64)&v129);
            v121 = v80;
            v65 = v79;
            sub_17CD270(&v125);
          }
          else
          {
            v127 = 0u;
            v128 = 0;
            v125 = v62;
            if ( v62 )
            {
              v108 = v57;
              sub_1F6CA20(&v125);
              v57 = v108;
            }
            v126 = *(_DWORD *)(a2 + 64);
            v63 = sub_1D2B730(
                    v116,
                    **(unsigned __int8 **)(a2 + 40),
                    *(_QWORD *)(*(_QWORD *)(a2 + 40) + 8LL),
                    (__int64)&v125,
                    *(_QWORD *)v112,
                    *(_QWORD *)(v112 + 8),
                    (__int64)v120,
                    v122,
                    v127,
                    v128,
                    v57,
                    v58,
                    (__int64)&v129,
                    0);
            v121 = v64;
            v65 = v63;
            sub_17CD270(&v125);
          }
          v129 = v65;
          LODWORD(v130) = v121;
          sub_1F994A0((__int64)a1, a2, &v129, 1, 1);
          sub_1F9A400((__int64)a1, a3, v65, 0, v65, 1, 1);
          sub_1F9A400((__int64)a1, a5, v65, 0, v65, 1, 1);
          LODWORD(v19) = v110;
        }
      }
    }
  }
  return (unsigned int)v19;
}
