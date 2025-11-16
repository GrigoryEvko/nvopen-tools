// Function: sub_2EF1130
// Address: 0x2ef1130
//
void __fastcall sub_2EF1130(__int64 a1, __int64 a2, _QWORD *a3, int a4, __int64 a5, __int64 a6)
{
  unsigned int *v9; // rdi
  __int64 v10; // rax
  __int64 v11; // r13
  __int64 v12; // rax
  unsigned __int64 v13; // r9
  __int64 v14; // r11
  unsigned __int64 v15; // r14
  __int64 v16; // r10
  __int64 *v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // r14
  unsigned __int64 v21; // rax
  _QWORD *v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  unsigned __int64 *v26; // rax
  __int64 v27; // r14
  __int64 *v28; // rdx
  __int64 v29; // r8
  __int64 v30; // r9
  unsigned __int64 v31; // r13
  __int64 v32; // r12
  unsigned __int64 v33; // r14
  unsigned int *v34; // r14
  __int64 v35; // rdi
  __int64 v36; // rdx
  __int64 v37; // rax
  void *v38; // rdx
  __int64 v39; // r8
  unsigned __int64 v40; // rsi
  _QWORD *v41; // rdi
  __int64 v42; // rdx
  __int64 v43; // r8
  _BYTE *v44; // rax
  __int64 v45; // r8
  _QWORD *v46; // rdx
  __int64 v47; // rax
  void *v48; // rdx
  __int64 v49; // r12
  _BYTE *v50; // rax
  _BYTE *v51; // rax
  unsigned __int64 v52; // rdx
  unsigned __int64 v53; // r14
  __int64 i; // r14
  int v55; // eax
  _QWORD *v56; // rax
  _QWORD *v57; // rdx
  __int64 v58; // rax
  __int64 m; // r14
  __int64 v60; // rdi
  unsigned __int64 j; // rax
  __int64 k; // rsi
  __int16 v63; // dx
  unsigned int v64; // esi
  __int64 v65; // r8
  unsigned int v66; // ecx
  __int64 *v67; // rdx
  __int64 v68; // r10
  __int64 v69; // rax
  __int64 v70; // r12
  unsigned __int64 v71; // rdx
  unsigned int v72; // eax
  __int64 v73; // r13
  unsigned int v74; // eax
  __int64 v75; // rcx
  __int64 *v76; // r14
  __int64 v77; // rax
  __int64 v78; // r14
  void *v79; // rdx
  _BYTE *v80; // rax
  __m128i *v81; // rdx
  __m128i si128; // xmm0
  _BYTE *v83; // rax
  unsigned __int64 v84; // rax
  __int64 v85; // rcx
  __int64 v86; // rsi
  __int64 v87; // rdx
  char v88; // r11
  bool v89; // r13
  __int64 v90; // rsi
  __int64 v91; // rdi
  int v92; // edx
  unsigned __int8 v93; // di
  unsigned __int16 v94; // r10
  __int64 *v95; // r10
  __int64 v96; // r8
  __int64 v97; // r10
  __int64 v98; // rdx
  __int64 v99; // rax
  unsigned __int64 v100; // rax
  char v101; // di
  __int64 v102; // r9
  unsigned __int64 v103; // r13
  __int64 *v104; // rax
  __int64 *v105; // rsi
  int v106; // r9d
  __int64 v107; // [rsp+8h] [rbp-108h]
  __int64 v108; // [rsp+8h] [rbp-108h]
  unsigned __int64 v109; // [rsp+10h] [rbp-100h]
  __int128 v110; // [rsp+18h] [rbp-F8h]
  unsigned __int64 *v111; // [rsp+28h] [rbp-E8h]
  char v112; // [rsp+28h] [rbp-E8h]
  bool v113; // [rsp+32h] [rbp-DEh]
  unsigned __int16 v114; // [rsp+32h] [rbp-DEh]
  unsigned int *v116; // [rsp+38h] [rbp-D8h]
  __int64 v117; // [rsp+40h] [rbp-D0h]
  __int64 v118; // [rsp+40h] [rbp-D0h]
  unsigned __int64 v119; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v120; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v121; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v122; // [rsp+48h] [rbp-C8h]
  unsigned __int64 *v123; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v124; // [rsp+48h] [rbp-C8h]
  char v125; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v126; // [rsp+48h] [rbp-C8h]
  unsigned __int64 v127; // [rsp+48h] [rbp-C8h]
  __int64 v128; // [rsp+48h] [rbp-C8h]
  _QWORD v129[2]; // [rsp+50h] [rbp-C0h] BYREF
  void (__fastcall *v130)(_QWORD *, _QWORD *, __int64); // [rsp+60h] [rbp-B0h]
  void (__fastcall *v131)(_QWORD *, __int64); // [rsp+68h] [rbp-A8h]
  _QWORD v132[2]; // [rsp+70h] [rbp-A0h] BYREF
  void (__fastcall *v133)(_QWORD *, _QWORD *, __int64); // [rsp+80h] [rbp-90h]
  void (__fastcall *v134)(_QWORD *, __int64); // [rsp+88h] [rbp-88h]
  _QWORD v135[2]; // [rsp+90h] [rbp-80h] BYREF
  void (__fastcall *v136)(_QWORD *, _QWORD *, __int64); // [rsp+A0h] [rbp-70h]
  void (__fastcall *v137)(_QWORD *, __int64); // [rsp+A8h] [rbp-68h]
  __int64 *v138; // [rsp+B0h] [rbp-60h] BYREF
  __int64 v139; // [rsp+B8h] [rbp-58h]
  _BYTE v140[80]; // [rsp+C0h] [rbp-50h] BYREF

  v9 = (unsigned int *)a3[2];
  *((_QWORD *)&v110 + 1) = a5;
  *(_QWORD *)&v110 = a6;
  v10 = *v9;
  v116 = v9;
  if ( (unsigned int)v10 >= *(_DWORD *)(a2 + 72) || v9 != *(unsigned int **)(*(_QWORD *)(a2 + 64) + 8 * v10) )
  {
    sub_2EEFF60(a1, "Foreign valno in live segment", *(__int64 **)(a1 + 32));
    sub_2EEFB40(a1, a2, a4, *((__int64 *)&v110 + 1), v110);
    sub_2EEF5A0(*(_QWORD *)(a1 + 16), (__int64)a3);
    sub_2EEF900(*(_QWORD *)(a1 + 16), v9);
  }
  if ( (*((_QWORD *)v9 + 1) & 0xFFFFFFFFFFFFFFF8LL) == 0 )
  {
    sub_2EEFF60(a1, "Live segment valno is marked unused", *(__int64 **)(a1 + 32));
    sub_2EEFB40(a1, a2, a4, *((__int64 *)&v110 + 1), v110);
    sub_2EEF5A0(*(_QWORD *)(a1 + 16), (__int64)a3);
  }
  v11 = *(_QWORD *)(*(_QWORD *)(a1 + 640) + 32LL);
  v12 = *(_QWORD *)((*a3 & 0xFFFFFFFFFFFFFFF8LL) + 16);
  if ( v12 )
  {
    v13 = *(_QWORD *)(v12 + 24);
  }
  else
  {
    v138 = (__int64 *)*a3;
    v13 = *(sub_2EEE710(
              *(_QWORD **)(v11 + 296),
              *(_QWORD *)(v11 + 296) + 16LL * *(unsigned int *)(v11 + 304),
              (__int64 *)&v138)
          - 1);
  }
  if ( !v13 )
  {
    sub_2EEFF60(a1, "Bad start of live segment, no basic block", *(__int64 **)(a1 + 32));
    goto LABEL_165;
  }
  if ( *a3 != *(_QWORD *)(*(_QWORD *)(v11 + 152) + 16LL * *(unsigned int *)(v13 + 24)) && *a3 != *((_QWORD *)v9 + 1) )
  {
    v122 = v13;
    sub_2EF03A0(a1, "Live segment must begin at MBB entry or valno def", v13);
    sub_2EEFB40(a1, a2, a4, *((__int64 *)&v110 + 1), v110);
    sub_2EEF5A0(*(_QWORD *)(a1 + 16), (__int64)a3);
    v13 = v122;
    v11 = *(_QWORD *)(*(_QWORD *)(a1 + 640) + 32LL);
  }
  v14 = a3[1];
  v15 = v14 & 0xFFFFFFFFFFFFFFF8LL;
  v16 = (v14 >> 1) & 3;
  if ( ((v14 >> 1) & 3) != 0 )
  {
    v17 = (__int64 *)(v15 | (2LL * ((int)v16 - 1)));
    v18 = *(_QWORD *)((a3[1] & 0xFFFFFFFFFFFFFFF8LL | (2LL * ((int)v16 - 1)) & 0xFFFFFFFFFFFFFFF8LL) + 0x10);
    if ( v18 )
    {
LABEL_14:
      v109 = *(_QWORD *)(v18 + 24);
      goto LABEL_15;
    }
  }
  else
  {
    v21 = *(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL;
    v18 = *(_QWORD *)(v21 + 16);
    v17 = (__int64 *)(v21 | 6);
    if ( v18 )
      goto LABEL_14;
  }
  v138 = v17;
  v121 = v13;
  v22 = sub_2EEE710(
          *(_QWORD **)(v11 + 296),
          *(_QWORD *)(v11 + 296) + 16LL * *(unsigned int *)(v11 + 304),
          (__int64 *)&v138);
  v13 = v121;
  v109 = *(v22 - 1);
LABEL_15:
  if ( !v109 )
  {
    sub_2EEFF60(a1, "Bad end of live segment, no basic block", *(__int64 **)(a1 + 32));
    goto LABEL_165;
  }
  if ( v14 == *(_QWORD *)(*(_QWORD *)(v11 + 152) + 16LL * *(unsigned int *)(v109 + 24) + 8) )
    goto LABEL_32;
  if ( a4 >= 0 )
  {
    v19 = *((_QWORD *)v9 + 1);
    if ( (v19 & 6) == 0 && *a3 == v19 && (v19 & 0xFFFFFFFFFFFFFFF8LL | 6) == v14 )
      return;
  }
  if ( !v16 )
  {
    v20 = *(_QWORD *)((*(_QWORD *)v15 & 0xFFFFFFFFFFFFFFF8LL) + 16);
    if ( v20 )
    {
      v119 = v13;
      sub_2EF03A0(a1, "Live segment ends at B slot of an instruction", v109);
      sub_2EEFB40(a1, a2, a4, *((__int64 *)&v110 + 1), v110);
      sub_2EEF5A0(*(_QWORD *)(a1 + 16), (__int64)a3);
      v14 = a3[1];
      v13 = v119;
      if ( ((v14 >> 1) & 3) != 3 )
        goto LABEL_30;
      goto LABEL_24;
    }
    goto LABEL_164;
  }
  v20 = *(_QWORD *)((((2LL * ((int)v16 - 1)) | v15) & 0xFFFFFFFFFFFFFFF8LL) + 0x10);
  if ( !v20 )
  {
LABEL_164:
    sub_2EF03A0(a1, "Live segment doesn't end at a valid instruction", v109);
LABEL_165:
    sub_2EEFB40(a1, a2, a4, *((__int64 *)&v110 + 1), v110);
    sub_2EEF5A0(*(_QWORD *)(a1 + 16), (__int64)a3);
    return;
  }
  if ( v16 != 3 )
    goto LABEL_30;
LABEL_24:
  if ( (*a3 & 0xFFFFFFFFFFFFFFF8LL) == (v14 & 0xFFFFFFFFFFFFFFF8LL) )
    goto LABEL_31;
  v120 = v13;
  sub_2EF03A0(a1, "Live segment ending at dead slot spans instructions", v109);
  sub_2EEFB40(a1, a2, a4, *((__int64 *)&v110 + 1), v110);
  sub_2EEF5A0(*(_QWORD *)(a1 + 16), (__int64)a3);
  v13 = v120;
LABEL_30:
  if ( (*(_BYTE *)(*(_QWORD *)(a1 + 32) + 345LL) & 1) != 0
    && (((__int64)a3[1] >> 1) & 3) == 1
    && (a3 + 3 == (_QWORD *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8)) || a3[1] != a3[3]) )
  {
    v124 = v13;
    sub_2EF03A0(
      a1,
      "Live segment ending at early clobber slot must be redefined by an EC def in the same instruction",
      v109);
    sub_2EEFB40(a1, a2, a4, *((__int64 *)&v110 + 1), v110);
    sub_2EEF5A0(*(_QWORD *)(a1 + 16), (__int64)a3);
    v13 = v124;
  }
LABEL_31:
  if ( a4 >= 0 )
    goto LABEL_32;
  v84 = v20;
  if ( (*(_BYTE *)(v20 + 44) & 4) != 0 )
  {
    do
      v84 = *(_QWORD *)v84 & 0xFFFFFFFFFFFFFFF8LL;
    while ( (*(_BYTE *)(v84 + 44) & 4) != 0 );
  }
  v85 = *(_QWORD *)(v20 + 24) + 48LL;
  do
  {
    v86 = *(_QWORD *)(v84 + 32);
    v87 = v86 + 40LL * (*(_DWORD *)(v84 + 40) & 0xFFFFFF);
    if ( v86 != v87 )
      goto LABEL_153;
    v84 = *(_QWORD *)(v84 + 8);
  }
  while ( v85 != v84 && (*(_BYTE *)(v84 + 44) & 4) != 0 );
  v84 = *(_QWORD *)(v20 + 24) + 48LL;
  if ( v86 == v87 )
  {
    if ( ((*((_BYTE *)a3 + 8) ^ 6) & 6) != 0 )
    {
      v125 = 0;
      goto LABEL_184;
    }
    v88 = 0;
LABEL_217:
    if ( *((_QWORD *)&v110 + 1) )
    {
      v117 = v13;
      v69 = *((_QWORD *)v116 + 1);
      if ( *a3 != v69 || (v69 & 6) == 0 )
      {
        v138 = (__int64 *)v140;
        v139 = 0x400000000LL;
        goto LABEL_123;
      }
      goto LABEL_130;
    }
    if ( !(_QWORD)v110 && !v88 )
    {
      v127 = v13;
      sub_2EF06E0(a1, "Instruction ending live segment on dead slot has no dead flag", v20);
      sub_2EEFB40(a1, a2, a4, 0, 0);
      sub_2EEF5A0(*(_QWORD *)(a1 + 16), (__int64)a3);
      v13 = v127;
    }
LABEL_191:
    v117 = v13;
    v23 = *((_QWORD *)v116 + 1);
    if ( *a3 == v23 )
      goto LABEL_129;
    v138 = (__int64 *)v140;
    v139 = 0x400000000LL;
LABEL_34:
    if ( !(_QWORD)v110 )
      goto LABEL_35;
LABEL_123:
    v70 = *(_QWORD *)(a1 + 640);
    v71 = *(unsigned int *)(v70 + 160);
    v72 = a4 & 0x7FFFFFFF;
    if ( (a4 & 0x7FFFFFFFu) < (unsigned int)v71 )
    {
      v73 = *(_QWORD *)(*(_QWORD *)(v70 + 152) + 8LL * v72);
      if ( v73 )
      {
LABEL_128:
        sub_2E0B070(v73, (__int64)&v138, *((__int64 *)&v110 + 1), v110, *(_QWORD **)(a1 + 64), *(_QWORD *)(a1 + 656));
LABEL_35:
        if ( a4 < 0 )
          goto LABEL_39;
LABEL_36:
        if ( !*(_BYTE *)(v117 + 216) )
        {
LABEL_39:
          while ( 1 )
          {
            v25 = *((_QWORD *)v116 + 1);
            v113 = (v25 & 6) == 0
                && *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 640) + 32LL) + 152LL)
                             + 16LL * *(unsigned int *)(v117 + 24)) == v25;
            v26 = *(unsigned __int64 **)(v117 + 64);
            v123 = v26;
            v111 = &v26[*(unsigned int *)(v117 + 72)];
            if ( v26 != v111 )
              break;
LABEL_106:
            if ( v117 == v109 )
              goto LABEL_117;
            v24 = v117;
LABEL_38:
            v117 = *(_QWORD *)(v24 + 8);
            if ( a4 >= 0 )
              goto LABEL_36;
          }
LABEL_49:
          v31 = *v123;
          v32 = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 640) + 32LL) + 152LL)
                          + 16LL * *(unsigned int *)(*v123 + 24)
                          + 8);
          if ( !*(_BYTE *)(v117 + 216) )
            goto LABEL_50;
          v52 = *(_QWORD *)(v31 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v52 )
LABEL_225:
            BUG();
          v53 = *(_QWORD *)(v31 + 48) & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_QWORD *)v52 & 4) == 0 && (*(_BYTE *)(v52 + 44) & 4) != 0 )
          {
            for ( i = *(_QWORD *)v52; ; i = *(_QWORD *)v53 )
            {
              v53 = i & 0xFFFFFFFFFFFFFFF8LL;
              if ( (*(_BYTE *)(v53 + 44) & 4) == 0 )
                break;
            }
          }
          if ( v53 == v31 + 48 )
          {
LABEL_50:
            v33 = v32 & 0xFFFFFFFFFFFFFFF8LL;
            goto LABEL_51;
          }
          while ( 1 )
          {
            v55 = *(_DWORD *)(v53 + 44);
            if ( (v55 & 4) != 0 || (v55 & 8) == 0 )
            {
              if ( (*(_QWORD *)(*(_QWORD *)(v53 + 16) + 24LL) & 0x80u) != 0LL )
              {
LABEL_95:
                v60 = *(_QWORD *)(a1 + 656);
                for ( j = v53; (*(_BYTE *)(j + 44) & 4) != 0; j = *(_QWORD *)j & 0xFFFFFFFFFFFFFFF8LL )
                  ;
                for ( ; (*(_BYTE *)(v53 + 44) & 8) != 0; v53 = *(_QWORD *)(v53 + 8) )
                  ;
                for ( k = *(_QWORD *)(v53 + 8); k != j; j = *(_QWORD *)(j + 8) )
                {
                  v63 = *(_WORD *)(j + 68);
                  if ( (unsigned __int16)(v63 - 14) > 4u && v63 != 24 )
                    break;
                }
                v64 = *(_DWORD *)(v60 + 144);
                v65 = *(_QWORD *)(v60 + 128);
                if ( v64 )
                {
                  v66 = (v64 - 1) & (((unsigned int)j >> 9) ^ ((unsigned int)j >> 4));
                  v67 = (__int64 *)(v65 + 16LL * v66);
                  v68 = *v67;
                  if ( *v67 == j )
                    goto LABEL_105;
                  v92 = 1;
                  while ( v68 != -4096 )
                  {
                    v106 = v92 + 1;
                    v66 = (v64 - 1) & (v92 + v66);
                    v67 = (__int64 *)(v65 + 16LL * v66);
                    v68 = *v67;
                    if ( *v67 == j )
                      goto LABEL_105;
                    v92 = v106;
                  }
                }
                v67 = (__int64 *)(v65 + 16LL * v64);
LABEL_105:
                v33 = v67[1] & 0xFFFFFFFFFFFFFFF8LL;
                v32 = v33 | 6;
LABEL_51:
                if ( ((v32 >> 1) & 3) != 0 )
                  v27 = (2LL * (int)(((v32 >> 1) & 3) - 1)) | v33;
                else
                  v27 = *(_QWORD *)v33 & 0xFFFFFFFFFFFFFFF8LL | 6;
                v28 = (__int64 *)sub_2E09D00((__int64 *)a2, v27);
                if ( v28 != (__int64 *)(*(_QWORD *)a2 + 24LL * *(unsigned int *)(a2 + 8))
                  && (*(_DWORD *)((*v28 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(*v28 >> 1) & 3) <= (*(_DWORD *)((v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v27 >> 1) & 3) )
                {
                  v34 = (unsigned int *)v28[2];
                  if ( v34 )
                  {
                    if ( v113 || v116 == v34 )
                      goto LABEL_48;
                    sub_2EF03A0(a1, "Different value live out of predecessor", v31);
                    sub_2EEFB40(a1, a2, a4, *((__int64 *)&v110 + 1), v110);
                    v35 = *(_QWORD *)(a1 + 16);
                    v36 = *(_QWORD *)(v35 + 32);
                    if ( (unsigned __int64)(*(_QWORD *)(v35 + 24) - v36) <= 6 )
                    {
                      v35 = sub_CB6200(v35, (unsigned __int8 *)"Valno #", 7u);
                    }
                    else
                    {
                      *(_DWORD *)v36 = 1852596566;
                      *(_WORD *)(v36 + 4) = 8303;
                      *(_BYTE *)(v36 + 6) = 35;
                      *(_QWORD *)(v35 + 32) += 7LL;
                    }
                    v37 = sub_CB59D0(v35, *v34);
                    v38 = *(void **)(v37 + 32);
                    v39 = v37;
                    if ( *(_QWORD *)(v37 + 24) - (_QWORD)v38 <= 0xCu )
                    {
                      v39 = sub_CB6200(v37, " live out of ", 0xDu);
                    }
                    else
                    {
                      qmemcpy(v38, " live out of ", 13);
                      *(_QWORD *)(v37 + 32) += 13LL;
                    }
                    v40 = v31;
                    v107 = v39;
                    v41 = v132;
                    sub_2E31000(v132, v31);
                    if ( v133 )
                    {
                      v134(v132, v107);
                      v43 = v107;
                      v44 = *(_BYTE **)(v107 + 32);
                      if ( (unsigned __int64)v44 >= *(_QWORD *)(v107 + 24) )
                      {
                        v43 = sub_CB5D20(v107, 64);
                      }
                      else
                      {
                        *(_QWORD *)(v107 + 32) = v44 + 1;
                        *v44 = 64;
                      }
                      v108 = v43;
                      v135[0] = v32;
                      sub_2FAD600(v135, v43);
                      v45 = v108;
                      v46 = *(_QWORD **)(v108 + 32);
                      if ( *(_QWORD *)(v108 + 24) - (_QWORD)v46 <= 7u )
                      {
                        v45 = sub_CB6200(v108, "\nValno #", 8u);
                      }
                      else
                      {
                        *v46 = 0x23206F6E6C61560ALL;
                        *(_QWORD *)(v108 + 32) += 8LL;
                      }
                      v47 = sub_CB59D0(v45, *v116);
                      v48 = *(void **)(v47 + 32);
                      v49 = v47;
                      if ( *(_QWORD *)(v47 + 24) - (_QWORD)v48 <= 0xAu )
                      {
                        v49 = sub_CB6200(v47, " live into ", 0xBu);
                      }
                      else
                      {
                        qmemcpy(v48, " live into ", 11);
                        *(_QWORD *)(v47 + 32) += 11LL;
                      }
                      v40 = v117;
                      v41 = v135;
                      sub_2E31000(v135, v117);
                      if ( v136 )
                      {
                        v137(v135, v49);
                        v50 = *(_BYTE **)(v49 + 32);
                        if ( (unsigned __int64)v50 >= *(_QWORD *)(v49 + 24) )
                        {
                          v49 = sub_CB5D20(v49, 64);
                        }
                        else
                        {
                          *(_QWORD *)(v49 + 32) = v50 + 1;
                          *v50 = 64;
                        }
                        v129[0] = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 640) + 32LL) + 152LL)
                                            + 16LL * *(unsigned int *)(v117 + 24));
                        sub_2FAD600(v129, v49);
                        v51 = *(_BYTE **)(v49 + 32);
                        if ( (unsigned __int64)v51 >= *(_QWORD *)(v49 + 24) )
                        {
                          sub_CB5D20(v49, 10);
                        }
                        else
                        {
                          *(_QWORD *)(v49 + 32) = v51 + 1;
                          *v51 = 10;
                        }
                        if ( v136 )
                          v136(v135, v135, 3);
                        if ( v133 )
                          v133(v132, v132, 3);
                        goto LABEL_48;
                      }
                    }
LABEL_222:
                    sub_4263D6(v41, v40, v42);
                  }
                }
                if ( v110 != 0 && v113
                  || (unsigned __int8)sub_2E1EC70(v31, v138, (unsigned int)v139, *(_QWORD *)(a1 + 656), v29, v30) )
                {
                  goto LABEL_48;
                }
                sub_2EF03A0(a1, "Register not marked live out of predecessor", v31);
                sub_2EEFB40(a1, a2, a4, *((__int64 *)&v110 + 1), v110);
                sub_2EEF900(*(_QWORD *)(a1 + 16), v116);
                v78 = *(_QWORD *)(a1 + 16);
                v79 = *(void **)(v78 + 32);
                if ( *(_QWORD *)(v78 + 24) - (_QWORD)v79 <= 0xAu )
                {
                  v78 = sub_CB6200(*(_QWORD *)(a1 + 16), " live into ", 0xBu);
                }
                else
                {
                  qmemcpy(v79, " live into ", 11);
                  *(_QWORD *)(v78 + 32) += 11LL;
                }
                v40 = v117;
                v41 = v129;
                sub_2E31000(v129, v117);
                if ( !v130 )
                  goto LABEL_222;
                v131(v129, v78);
                v80 = *(_BYTE **)(v78 + 32);
                if ( (unsigned __int64)v80 >= *(_QWORD *)(v78 + 24) )
                {
                  v78 = sub_CB5D20(v78, 64);
                }
                else
                {
                  *(_QWORD *)(v78 + 32) = v80 + 1;
                  *v80 = 64;
                }
                v135[0] = *(_QWORD *)(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 640) + 32LL) + 152LL)
                                    + 16LL * *(unsigned int *)(v117 + 24));
                sub_2FAD600(v135, v78);
                v81 = *(__m128i **)(v78 + 32);
                if ( *(_QWORD *)(v78 + 24) - (_QWORD)v81 <= 0x11u )
                {
                  v78 = sub_CB6200(v78, ", not live before ", 0x12u);
                }
                else
                {
                  si128 = _mm_load_si128((const __m128i *)&xmmword_4453DB0);
                  v81[1].m128i_i16[0] = 8293;
                  *v81 = si128;
                  *(_QWORD *)(v78 + 32) += 18LL;
                }
                v135[0] = v32;
                sub_2FAD600(v135, v78);
                v83 = *(_BYTE **)(v78 + 32);
                if ( (unsigned __int64)v83 >= *(_QWORD *)(v78 + 24) )
                {
                  sub_CB5D20(v78, 10);
                }
                else
                {
                  *(_QWORD *)(v78 + 32) = v83 + 1;
                  *v83 = 10;
                }
                if ( v130 )
                  v130(v129, v129, 3);
LABEL_48:
                if ( v111 == ++v123 )
                  goto LABEL_106;
                goto LABEL_49;
              }
            }
            else if ( sub_2E88A90(v53, 128, 1) )
            {
              goto LABEL_95;
            }
            v56 = (_QWORD *)(*(_QWORD *)v53 & 0xFFFFFFFFFFFFFFF8LL);
            v57 = v56;
            if ( !v56 )
              goto LABEL_225;
            v53 = *(_QWORD *)v53 & 0xFFFFFFFFFFFFFFF8LL;
            v58 = *v56;
            if ( (v58 & 4) == 0 && (*((_BYTE *)v57 + 44) & 4) != 0 )
            {
              for ( m = v58; ; m = *(_QWORD *)v53 )
              {
                v53 = m & 0xFFFFFFFFFFFFFFF8LL;
                if ( (*(_BYTE *)(v53 + 44) & 4) == 0 )
                  break;
              }
            }
            if ( v53 == v31 + 48 )
              goto LABEL_50;
          }
        }
        v24 = v117;
        if ( v117 != v109 )
          goto LABEL_38;
LABEL_117:
        if ( v138 != (__int64 *)v140 )
          _libc_free((unsigned __int64)v138);
        return;
      }
    }
    v74 = v72 + 1;
    if ( (unsigned int)v71 < v74 && v74 != v71 )
    {
      if ( v74 >= v71 )
      {
        v102 = *(_QWORD *)(v70 + 168);
        v103 = v74 - v71;
        if ( v74 > (unsigned __int64)*(unsigned int *)(v70 + 164) )
        {
          v128 = *(_QWORD *)(v70 + 168);
          sub_C8D5F0(v70 + 152, (const void *)(v70 + 168), v74, 8u, v74, v102);
          v71 = *(unsigned int *)(v70 + 160);
          v102 = v128;
        }
        v75 = *(_QWORD *)(v70 + 152);
        v104 = (__int64 *)(v75 + 8 * v71);
        v105 = &v104[v103];
        if ( v104 != v105 )
        {
          do
            *v104++ = v102;
          while ( v105 != v104 );
          LODWORD(v71) = *(_DWORD *)(v70 + 160);
          v75 = *(_QWORD *)(v70 + 152);
        }
        *(_DWORD *)(v70 + 160) = v71 + v103;
        goto LABEL_127;
      }
      *(_DWORD *)(v70 + 160) = v74;
    }
    v75 = *(_QWORD *)(v70 + 152);
LABEL_127:
    v76 = (__int64 *)(v75 + 8LL * (a4 & 0x7FFFFFFF));
    v77 = sub_2E10F30(a4);
    *v76 = v77;
    v73 = v77;
    sub_2E11E80((_QWORD *)v70, v77);
    goto LABEL_128;
  }
LABEL_153:
  v88 = 0;
  v89 = 0;
  v125 = 0;
  while ( 1 )
  {
    if ( *(_BYTE *)v86 || *(_DWORD *)(v86 + 8) != a4 )
      goto LABEL_156;
    v93 = *(_BYTE *)(v86 + 3);
    v94 = (*(_DWORD *)v86 >> 8) & 0xFFF;
    v114 = v94;
    if ( v94 )
    {
      v95 = (__int64 *)(*(_QWORD *)(*(_QWORD *)(a1 + 56) + 272LL) + 16LL * v94);
      v96 = *v95;
      v97 = v95[1];
      v118 = v96;
      v112 = v93 & 0x10;
      if ( (v93 & 0x10) != 0 )
      {
        v118 = ~v96;
        v97 = ~v97;
        v125 = 1;
LABEL_195:
        v101 = (v93 >> 6) & ((v93 & 0x10) != 0);
        if ( v101 )
          v88 = v101;
      }
    }
    else
    {
      v97 = -1;
      v118 = -1;
      v112 = v93 & 0x10;
      if ( (v93 & 0x10) != 0 )
        goto LABEL_195;
    }
    if ( (v110 == 0 || *((_QWORD *)&v110 + 1) & v118 | (unsigned __int64)v110 & v97)
      && (*(_BYTE *)(v86 + 4) & 1) == 0
      && (*(_BYTE *)(v86 + 4) & 2) == 0
      && (v112 == 0 || v114 != 0) )
    {
      v89 = v112 == 0 || v114 != 0;
    }
LABEL_156:
    v90 = v86 + 40;
    v91 = v87;
    if ( v90 != v87 )
    {
      v87 = v90;
      goto LABEL_163;
    }
    while ( 1 )
    {
      v84 = *(_QWORD *)(v84 + 8);
      if ( v85 == v84 || (*(_BYTE *)(v84 + 44) & 4) == 0 )
        break;
      v87 = *(_QWORD *)(v84 + 32);
      v91 = v87 + 40LL * (*(_DWORD *)(v84 + 40) & 0xFFFFFF);
      if ( v87 != v91 )
        goto LABEL_163;
    }
    if ( v91 == v87 )
      break;
    v84 = *(_QWORD *)(v20 + 24) + 48LL;
LABEL_163:
    v86 = v87;
    v87 = v91;
  }
  if ( ((*((_BYTE *)a3 + 8) ^ 6) & 6) == 0 )
    goto LABEL_217;
  if ( !v89 )
  {
LABEL_184:
    v98 = *(_QWORD *)(a1 + 64);
    v99 = *(_QWORD *)(*(_QWORD *)(v98 + 56) + 16LL * (a4 & 0x7FFFFFFF));
    if ( v99 )
    {
      if ( (v99 & 4) == 0 )
      {
        v100 = v99 & 0xFFFFFFFFFFFFFFF8LL;
        if ( v100 )
        {
          if ( *(_BYTE *)(v98 + 48) && *(_BYTE *)(v100 + 43) && v110 == 0 && v125 )
            goto LABEL_191;
        }
      }
    }
    v126 = v13;
    sub_2EF06E0(a1, "Instruction ending live segment doesn't read the register", v20);
    sub_2EEFB40(a1, a2, a4, *((__int64 *)&v110 + 1), v110);
    sub_2EEF5A0(*(_QWORD *)(a1 + 16), (__int64)a3);
    v13 = v126;
  }
LABEL_32:
  v23 = *a3;
  v117 = v13;
  if ( *((_QWORD *)v116 + 1) != *a3 )
  {
LABEL_33:
    v138 = (__int64 *)v140;
    v139 = 0x400000000LL;
    if ( !*((_QWORD *)&v110 + 1) )
      goto LABEL_34;
    goto LABEL_123;
  }
LABEL_129:
  if ( (v23 & 6) == 0 )
    goto LABEL_33;
LABEL_130:
  if ( v13 != v109 )
  {
    v117 = *(_QWORD *)(v13 + 8);
    goto LABEL_33;
  }
}
