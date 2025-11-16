// Function: sub_D64DE0
// Address: 0xd64de0
//
__int64 __fastcall sub_D64DE0(__int64 a1, __int64 a2)
{
  int v4; // edx
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // r12
  int v9; // edx
  __int64 v10; // r13
  unsigned int v11; // esi
  __int64 v12; // rdi
  __int64 v13; // rcx
  _QWORD *v14; // r8
  unsigned int v15; // edx
  int v16; // r10d
  _QWORD *v17; // r14
  __int64 v18; // rax
  unsigned __int64 *v19; // rcx
  unsigned __int64 *v20; // r14
  unsigned __int64 v21; // rdx
  __int64 v22; // rax
  unsigned __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // r14
  __int64 v26; // rax
  __int16 v27; // dx
  __int64 v28; // rdi
  __int64 v29; // rsi
  __int64 v30; // r8
  unsigned __int64 v31; // rsi
  unsigned __int64 *v32; // rax
  int v33; // ecx
  unsigned __int64 *v34; // rdx
  __int64 v35; // rdx
  __int64 v36; // rcx
  int v37; // eax
  int v38; // eax
  unsigned int v39; // esi
  __int64 v40; // rax
  __int64 v41; // rsi
  __int64 v42; // rsi
  int v43; // eax
  int v44; // eax
  unsigned int v45; // ecx
  __int64 v46; // rax
  __int64 v47; // rcx
  __int64 v48; // rcx
  __int64 v49; // rax
  __int64 v50; // rbx
  _QWORD *v51; // rsi
  _QWORD *v52; // rdx
  _QWORD *v53; // rax
  __int64 v54; // rcx
  __int64 v55; // rax
  _QWORD *v56; // rsi
  _QWORD *v57; // rdx
  _QWORD *v58; // rax
  __int64 v59; // rcx
  __int64 v61; // rax
  _QWORD *v62; // rsi
  _QWORD *v63; // rdx
  _QWORD *v64; // rax
  __int64 v65; // rcx
  __int64 v66; // rax
  _QWORD *v67; // rsi
  _QWORD *v68; // rdx
  _QWORD *v69; // rax
  __int64 v70; // rcx
  unsigned __int64 v71; // rdi
  __int64 *v72; // rax
  __int64 *v73; // rax
  int v74; // eax
  int v75; // eax
  __int64 v76; // rax
  bool v77; // zf
  __int64 v78; // rax
  unsigned __int64 v79; // rsi
  unsigned __int64 v80; // rax
  int v81; // ecx
  int v82; // ecx
  __int64 v83; // rdi
  __int64 v84; // rdx
  __int64 v85; // rsi
  int v86; // r11d
  _QWORD *v87; // r9
  int v88; // edx
  int v89; // edx
  int v90; // r10d
  _QWORD *v91; // rsi
  __int64 v92; // rdi
  __int64 v93; // r11
  __int64 v94; // rcx
  __int64 *v95; // rax
  __int64 *v96; // rax
  __int64 v97; // [rsp+8h] [rbp-F8h]
  __int64 v98; // [rsp+10h] [rbp-F0h]
  unsigned __int64 v99; // [rsp+10h] [rbp-F0h]
  __int64 v100; // [rsp+18h] [rbp-E8h]
  __int64 v101; // [rsp+18h] [rbp-E8h]
  __int64 v102; // [rsp+18h] [rbp-E8h]
  __int64 v103; // [rsp+20h] [rbp-E0h]
  __int64 v104; // [rsp+28h] [rbp-D8h]
  _QWORD *v105; // [rsp+28h] [rbp-D8h]
  unsigned __int64 *v106; // [rsp+38h] [rbp-C8h]
  unsigned __int64 *v107; // [rsp+38h] [rbp-C8h]
  unsigned __int64 *v108; // [rsp+38h] [rbp-C8h]
  __int64 v109; // [rsp+38h] [rbp-C8h]
  unsigned __int64 *v110; // [rsp+38h] [rbp-C8h]
  unsigned __int64 *v111; // [rsp+38h] [rbp-C8h]
  unsigned __int64 *v112; // [rsp+38h] [rbp-C8h]
  unsigned __int64 *v113; // [rsp+38h] [rbp-C8h]
  unsigned int v114; // [rsp+38h] [rbp-C8h]
  __int64 v115; // [rsp+60h] [rbp-A0h] BYREF
  __int64 v116; // [rsp+68h] [rbp-98h]
  __int64 v117; // [rsp+70h] [rbp-90h]
  __int64 v118; // [rsp+80h] [rbp-80h] BYREF
  __int64 v119; // [rsp+88h] [rbp-78h]
  __int64 v120; // [rsp+90h] [rbp-70h]
  __int64 v121; // [rsp+A0h] [rbp-60h] BYREF
  __int64 v122; // [rsp+A8h] [rbp-58h]
  __int64 v123; // [rsp+B0h] [rbp-50h]
  unsigned __int64 v124; // [rsp+B8h] [rbp-48h] BYREF
  __int64 v125; // [rsp+C0h] [rbp-40h]
  __int64 v126; // [rsp+C8h] [rbp-38h]

  v4 = *(_DWORD *)(a2 + 4);
  v5 = *(_QWORD *)(a1 + 208);
  LOWORD(v125) = 257;
  v103 = a1 + 24;
  v6 = sub_D5C860((__int64 *)(a1 + 24), v5, v4 & 0x7FFFFFF, (__int64)&v121);
  v7 = *(_QWORD *)(a1 + 208);
  v8 = v6;
  v9 = *(_DWORD *)(a2 + 4) & 0x7FFFFFF;
  LOWORD(v125) = 257;
  v118 = 6;
  v120 = sub_D5C860((__int64 *)(a1 + 24), v7, v9, (__int64)&v121);
  v10 = v120;
  v119 = 0;
  if ( v120 != -4096 && v120 != 0 && v120 != -8192 )
    sub_BD73F0((__int64)&v118);
  v117 = v8;
  v115 = 6;
  v116 = 0;
  if ( v8 == -4096 || v8 == 0 || v8 == -8192 )
  {
    v121 = 6;
    v122 = 0;
    v123 = v8;
  }
  else
  {
    sub_BD73F0((__int64)&v115);
    v121 = 6;
    v122 = 0;
    v123 = v117;
    if ( v117 != -4096 && v117 != 0 && v117 != -8192 )
      sub_BD6050((unsigned __int64 *)&v121, v115 & 0xFFFFFFFFFFFFFFF8LL);
  }
  v124 = 6;
  v125 = 0;
  v126 = v120;
  if ( v120 != 0 && v120 != -4096 && v120 != -8192 )
    sub_BD6050(&v124, v118 & 0xFFFFFFFFFFFFFFF8LL);
  if ( v117 != 0 && v117 != -4096 && v117 != -8192 )
    sub_BD60C0(&v115);
  if ( v120 != 0 && v120 != -4096 && v120 != -8192 )
    sub_BD60C0(&v118);
  v11 = *(_DWORD *)(a1 + 248);
  v12 = a1 + 224;
  if ( !v11 )
  {
    ++*(_QWORD *)(a1 + 224);
    goto LABEL_140;
  }
  v13 = *(_QWORD *)(a1 + 232);
  v14 = 0;
  v15 = (v11 - 1) & (((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9));
  v16 = 1;
  v17 = (_QWORD *)(v13 + 56LL * v15);
  v18 = *v17;
  if ( a2 == *v17 )
  {
LABEL_21:
    v19 = v17 + 1;
    v20 = v17 + 4;
    goto LABEL_22;
  }
  while ( v18 != -4096 )
  {
    if ( !v14 && v18 == -8192 )
      v14 = v17;
    v15 = (v11 - 1) & (v16 + v15);
    v17 = (_QWORD *)(v13 + 56LL * v15);
    v18 = *v17;
    if ( a2 == *v17 )
      goto LABEL_21;
    ++v16;
  }
  v74 = *(_DWORD *)(a1 + 240);
  if ( !v14 )
    v14 = v17;
  ++*(_QWORD *)(a1 + 224);
  v75 = v74 + 1;
  if ( 4 * v75 >= 3 * v11 )
  {
LABEL_140:
    sub_D5F740(v12, 2 * v11);
    v81 = *(_DWORD *)(a1 + 248);
    if ( v81 )
    {
      v82 = v81 - 1;
      v83 = *(_QWORD *)(a1 + 232);
      LODWORD(v84) = v82 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v14 = (_QWORD *)(v83 + 56LL * (unsigned int)v84);
      v85 = *v14;
      v75 = *(_DWORD *)(a1 + 240) + 1;
      if ( a2 != *v14 )
      {
        v86 = 1;
        v87 = 0;
        while ( v85 != -4096 )
        {
          if ( !v87 && v85 == -8192 )
            v87 = v14;
          v84 = v82 & (unsigned int)(v84 + v86);
          v14 = (_QWORD *)(v83 + 56 * v84);
          v85 = *v14;
          if ( a2 == *v14 )
            goto LABEL_122;
          ++v86;
        }
        if ( v87 )
          v14 = v87;
      }
      goto LABEL_122;
    }
    goto LABEL_169;
  }
  if ( v11 - *(_DWORD *)(a1 + 244) - v75 <= v11 >> 3 )
  {
    v114 = ((unsigned int)a2 >> 4) ^ ((unsigned int)a2 >> 9);
    sub_D5F740(v12, v11);
    v88 = *(_DWORD *)(a1 + 248);
    if ( v88 )
    {
      v89 = v88 - 1;
      v90 = 1;
      v91 = 0;
      v92 = *(_QWORD *)(a1 + 232);
      LODWORD(v93) = v89 & v114;
      v14 = (_QWORD *)(v92 + 56LL * (v89 & v114));
      v94 = *v14;
      v75 = *(_DWORD *)(a1 + 240) + 1;
      if ( a2 != *v14 )
      {
        while ( v94 != -4096 )
        {
          if ( !v91 && v94 == -8192 )
            v91 = v14;
          v93 = v89 & (unsigned int)(v93 + v90);
          v14 = (_QWORD *)(v92 + 56 * v93);
          v94 = *v14;
          if ( a2 == *v14 )
            goto LABEL_122;
          ++v90;
        }
        if ( v91 )
          v14 = v91;
      }
      goto LABEL_122;
    }
LABEL_169:
    ++*(_DWORD *)(a1 + 240);
    BUG();
  }
LABEL_122:
  *(_DWORD *)(a1 + 240) = v75;
  if ( *v14 != -4096 )
    --*(_DWORD *)(a1 + 244);
  *v14 = a2;
  v19 = v14 + 1;
  v117 = 0;
  v118 = 6;
  v119 = 0;
  v120 = 0;
  v115 = 6;
  v116 = 0;
  v14[1] = 6;
  v14[2] = 0;
  v76 = v117;
  v77 = v117 == 0;
  v14[3] = v117;
  if ( v76 != -4096 && !v77 && v76 != -8192 )
  {
    v105 = v14;
    v110 = v14 + 1;
    sub_BD6050(v14 + 1, 0);
    v14 = v105;
    v19 = v110;
  }
  v14[4] = 6;
  v20 = v14 + 4;
  v14[5] = 0;
  v78 = v120;
  v77 = v120 == 0;
  v14[6] = v120;
  if ( v78 != -4096 && !v77 && v78 != -8192 )
  {
    v111 = v19;
    sub_BD6050(v14 + 4, v118 & 0xFFFFFFFFFFFFFFF8LL);
    v19 = v111;
  }
  if ( v117 != 0 && v117 != -4096 && v117 != -8192 )
  {
    v112 = v19;
    sub_BD60C0(&v115);
    v19 = v112;
  }
  if ( v120 != 0 && v120 != -4096 && v120 != -8192 )
  {
    v113 = v19;
    sub_BD60C0(&v118);
    v19 = v113;
  }
LABEL_22:
  v21 = v19[2];
  v22 = v123;
  if ( v21 != v123 )
  {
    if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
    {
      v106 = v19;
      sub_BD60C0(v19);
      v22 = v123;
      v19 = v106;
    }
    v19[2] = v22;
    if ( v22 != 0 && v22 != -4096 && v22 != -8192 )
    {
      v107 = v19;
      sub_BD6050(v19, v121 & 0xFFFFFFFFFFFFFFF8LL);
      v19 = v107;
    }
  }
  v23 = v19[5];
  v24 = v126;
  if ( v23 != v126 )
  {
    if ( v23 != 0 && v23 != -4096 && v23 != -8192 )
    {
      v108 = v19;
      sub_BD60C0(v20);
      v24 = v126;
      v19 = v108;
    }
    v19[5] = v24;
    if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
      sub_BD6050(v20, v124 & 0xFFFFFFFFFFFFFFF8LL);
    v24 = v126;
  }
  if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
    sub_BD60C0(&v124);
  if ( v123 != -4096 && v123 != 0 && v123 != -8192 )
    sub_BD60C0(&v121);
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) == 0 )
  {
LABEL_72:
    v49 = sub_B48DC0(v8);
    v50 = v49;
    if ( v49 )
    {
      sub_BD84D0(v8, v49);
      sub_B43D60((_QWORD *)v8);
      if ( *(_BYTE *)(a1 + 396) )
      {
        v51 = *(_QWORD **)(a1 + 376);
        v52 = &v51[*(unsigned int *)(a1 + 388)];
        if ( v51 != v52 )
        {
          v53 = *(_QWORD **)(a1 + 376);
          while ( v8 != *v53 )
          {
            if ( v52 == ++v53 )
              goto LABEL_153;
          }
          v54 = (unsigned int)(*(_DWORD *)(a1 + 388) - 1);
          v8 = v50;
          *(_DWORD *)(a1 + 388) = v54;
          *v53 = v51[v54];
          ++*(_QWORD *)(a1 + 368);
          goto LABEL_79;
        }
      }
      else
      {
        v96 = sub_C8CA60(a1 + 368, v8);
        if ( v96 )
        {
          *v96 = -2;
          v8 = v50;
          ++*(_DWORD *)(a1 + 392);
          ++*(_QWORD *)(a1 + 368);
          goto LABEL_79;
        }
      }
LABEL_153:
      v8 = v50;
    }
LABEL_79:
    v55 = sub_B48DC0(v10);
    if ( v55 )
    {
      sub_BD84D0(v10, v55);
      sub_B43D60((_QWORD *)v10);
      if ( *(_BYTE *)(a1 + 396) )
      {
        v56 = *(_QWORD **)(a1 + 376);
        v57 = &v56[*(unsigned int *)(a1 + 388)];
        if ( v56 != v57 )
        {
          v58 = *(_QWORD **)(a1 + 376);
          while ( v10 != *v58 )
          {
            if ( v57 == ++v58 )
              return v8;
          }
          v59 = (unsigned int)(*(_DWORD *)(a1 + 388) - 1);
          *(_DWORD *)(a1 + 388) = v59;
          *v58 = v56[v59];
          ++*(_QWORD *)(a1 + 368);
        }
      }
      else
      {
        v95 = sub_C8CA60(a1 + 368, v10);
        if ( v95 )
        {
          *v95 = -2;
          ++*(_DWORD *)(a1 + 392);
          ++*(_QWORD *)(a1 + 368);
        }
      }
    }
    return v8;
  }
  v109 = 0;
  v104 = 8LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  while ( 1 )
  {
    v25 = *(_QWORD *)(*(_QWORD *)(a2 - 8) + 32LL * *(unsigned int *)(a2 + 72) + v109);
    v26 = sub_AA5190(v25);
    v28 = v26;
    if ( !v26 )
    {
      *(_QWORD *)(a1 + 72) = v25;
      *(_QWORD *)(a1 + 80) = 0;
      *(_WORD *)(a1 + 88) = 0;
LABEL_47:
      v29 = *(_QWORD *)sub_B46C60(v28);
      v121 = v29;
      if ( v29 && (sub_B96E90((__int64)&v121, v29, 1), (v30 = v121) != 0) )
      {
        v31 = *(unsigned int *)(a1 + 32);
        v32 = *(unsigned __int64 **)(a1 + 24);
        v33 = *(_DWORD *)(a1 + 32);
        v34 = &v32[2 * v31];
        if ( v32 != v34 )
        {
          while ( *(_DWORD *)v32 )
          {
            v32 += 2;
            if ( v34 == v32 )
              goto LABEL_103;
          }
          v32[1] = v121;
LABEL_54:
          sub_B91220((__int64)&v121, v30);
          goto LABEL_55;
        }
LABEL_103:
        v71 = *(unsigned int *)(a1 + 36);
        if ( v31 >= v71 )
        {
          v79 = v31 + 1;
          v80 = v97 & 0xFFFFFFFF00000000LL;
          v97 &= 0xFFFFFFFF00000000LL;
          if ( v71 < v79 )
          {
            v99 = v80;
            v102 = v121;
            sub_C8D5F0(v103, (const void *)(a1 + 40), v79, 0x10u, v121, a1 + 40);
            v80 = v99;
            v30 = v102;
            v34 = (unsigned __int64 *)(*(_QWORD *)(a1 + 24) + 16LL * *(unsigned int *)(a1 + 32));
          }
          *v34 = v80;
          v34[1] = v30;
          v30 = v121;
          ++*(_DWORD *)(a1 + 32);
        }
        else
        {
          if ( v34 )
          {
            *(_DWORD *)v34 = 0;
            v34[1] = v30;
            v33 = *(_DWORD *)(a1 + 32);
            v30 = v121;
          }
          *(_DWORD *)(a1 + 32) = v33 + 1;
        }
      }
      else
      {
        sub_93FB40(v103, 0);
        v30 = v121;
      }
      if ( !v30 )
        goto LABEL_55;
      goto LABEL_54;
    }
    *(_QWORD *)(a1 + 80) = v26;
    v28 = v26 - 24;
    *(_QWORD *)(a1 + 72) = v25;
    *(_WORD *)(a1 + 88) = v27;
    if ( v26 != v25 + 48 )
      goto LABEL_47;
LABEL_55:
    v36 = sub_D63080(a1, *(unsigned __int8 **)(*(_QWORD *)(a2 - 8) + 4 * v109));
    if ( !v36 || !v35 )
      break;
    v37 = *(_DWORD *)(v8 + 4) & 0x7FFFFFF;
    if ( v37 == *(_DWORD *)(v8 + 72) )
    {
      v98 = v35;
      v101 = v36;
      sub_B48D90(v8);
      v35 = v98;
      v36 = v101;
      v37 = *(_DWORD *)(v8 + 4) & 0x7FFFFFF;
    }
    v38 = (v37 + 1) & 0x7FFFFFF;
    v39 = v38 | *(_DWORD *)(v8 + 4) & 0xF8000000;
    v40 = *(_QWORD *)(v8 - 8) + 32LL * (unsigned int)(v38 - 1);
    *(_DWORD *)(v8 + 4) = v39;
    if ( *(_QWORD *)v40 )
    {
      v41 = *(_QWORD *)(v40 + 8);
      **(_QWORD **)(v40 + 16) = v41;
      if ( v41 )
        *(_QWORD *)(v41 + 16) = *(_QWORD *)(v40 + 16);
    }
    *(_QWORD *)v40 = v36;
    v42 = *(_QWORD *)(v36 + 16);
    *(_QWORD *)(v40 + 8) = v42;
    if ( v42 )
      *(_QWORD *)(v42 + 16) = v40 + 8;
    *(_QWORD *)(v40 + 16) = v36 + 16;
    *(_QWORD *)(v36 + 16) = v40;
    *(_QWORD *)(*(_QWORD *)(v8 - 8) + 32LL * *(unsigned int *)(v8 + 72) + 8LL * ((*(_DWORD *)(v8 + 4) & 0x7FFFFFFu) - 1)) = v25;
    v43 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
    if ( v43 == *(_DWORD *)(v10 + 72) )
    {
      v100 = v35;
      sub_B48D90(v10);
      v35 = v100;
      v43 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
    }
    v44 = (v43 + 1) & 0x7FFFFFF;
    v45 = v44 | *(_DWORD *)(v10 + 4) & 0xF8000000;
    v46 = *(_QWORD *)(v10 - 8) + 32LL * (unsigned int)(v44 - 1);
    *(_DWORD *)(v10 + 4) = v45;
    if ( *(_QWORD *)v46 )
    {
      v47 = *(_QWORD *)(v46 + 8);
      **(_QWORD **)(v46 + 16) = v47;
      if ( v47 )
        *(_QWORD *)(v47 + 16) = *(_QWORD *)(v46 + 16);
    }
    *(_QWORD *)v46 = v35;
    v48 = *(_QWORD *)(v35 + 16);
    *(_QWORD *)(v46 + 8) = v48;
    if ( v48 )
      *(_QWORD *)(v48 + 16) = v46 + 8;
    *(_QWORD *)(v46 + 16) = v35 + 16;
    *(_QWORD *)(v35 + 16) = v46;
    v109 += 8;
    *(_QWORD *)(*(_QWORD *)(v10 - 8)
              + 32LL * *(unsigned int *)(v10 + 72)
              + 8LL * ((*(_DWORD *)(v10 + 4) & 0x7FFFFFFu) - 1)) = v25;
    if ( v104 == v109 )
      goto LABEL_72;
  }
  v61 = sub_ACADE0(*(__int64 ***)(a1 + 208));
  sub_BD84D0(v10, v61);
  sub_B43D60((_QWORD *)v10);
  if ( *(_BYTE *)(a1 + 396) )
  {
    v62 = *(_QWORD **)(a1 + 376);
    v63 = &v62[*(unsigned int *)(a1 + 388)];
    v64 = v62;
    if ( v62 != v63 )
    {
      while ( v10 != *v64 )
      {
        if ( v63 == ++v64 )
          goto LABEL_93;
      }
      v65 = (unsigned int)(*(_DWORD *)(a1 + 388) - 1);
      *(_DWORD *)(a1 + 388) = v65;
      *v64 = v62[v65];
      ++*(_QWORD *)(a1 + 368);
    }
  }
  else
  {
    v73 = sub_C8CA60(a1 + 368, v10);
    if ( v73 )
    {
      *v73 = -2;
      ++*(_DWORD *)(a1 + 392);
      ++*(_QWORD *)(a1 + 368);
    }
  }
LABEL_93:
  v66 = sub_ACADE0(*(__int64 ***)(a1 + 208));
  sub_BD84D0(v8, v66);
  sub_B43D60((_QWORD *)v8);
  if ( *(_BYTE *)(a1 + 396) )
  {
    v67 = *(_QWORD **)(a1 + 376);
    v68 = &v67[*(unsigned int *)(a1 + 388)];
    v69 = v67;
    if ( v67 != v68 )
    {
      while ( v8 != *v69 )
      {
        if ( v68 == ++v69 )
          return 0;
      }
      v70 = (unsigned int)(*(_DWORD *)(a1 + 388) - 1);
      *(_DWORD *)(a1 + 388) = v70;
      *v69 = v67[v70];
      ++*(_QWORD *)(a1 + 368);
    }
  }
  else
  {
    v72 = sub_C8CA60(a1 + 368, v8);
    if ( v72 )
    {
      *v72 = -2;
      ++*(_DWORD *)(a1 + 392);
      ++*(_QWORD *)(a1 + 368);
    }
  }
  return 0;
}
