// Function: sub_6F0CB0
// Address: 0x6f0cb0
//
__int64 __fastcall sub_6F0CB0(
        __int64 a1,
        unsigned __int64 a2,
        __m128i *a3,
        __int64 a4,
        __int64 *a5,
        int *a6,
        unsigned int *a7)
{
  unsigned __int64 v7; // r15
  int *v8; // rax
  unsigned __int64 **v9; // r12
  char v10; // al
  __int64 v11; // rax
  _QWORD *v12; // r13
  __int64 v13; // rbx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 result; // rax
  char v17; // al
  __int64 v18; // r13
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rsi
  __m128i *v22; // r13
  __int64 *v23; // r11
  unsigned __int64 *v24; // r9
  __int64 v25; // r14
  unsigned __int64 *v26; // r10
  __int64 v27; // rdx
  __int64 v28; // rcx
  const __m128i *v29; // rbx
  const __m128i *v30; // rcx
  __m128i *v31; // rax
  __int64 v32; // rax
  __int64 v33; // rdx
  __int64 v34; // rcx
  __int64 v35; // r8
  char v36; // al
  __int64 v37; // r14
  __int64 v38; // r11
  __int64 v39; // rax
  char v40; // r10
  char v41; // r10
  __int64 v42; // rdx
  _DWORD *v43; // r12
  __int64 v44; // r12
  __int64 v45; // rdi
  unsigned int v46; // eax
  unsigned __int64 v47; // r14
  __int64 v48; // rbx
  _QWORD *v49; // rax
  unsigned int v50; // r12d
  __int64 v51; // r13
  int v52; // r14d
  int **v53; // rbx
  int *v54; // rax
  __int64 v55; // r10
  int v56; // eax
  __m128i *v57; // rax
  __int64 v58; // rdx
  unsigned __int64 *v59; // r10
  const __m128i *v60; // rdi
  __m128i *v61; // r8
  int v62; // eax
  __int64 v63; // r13
  __int64 v64; // rax
  int v65; // eax
  __int16 v66; // r14
  __int64 v67; // rax
  __int64 v68; // rcx
  __int64 v69; // r8
  __int64 v70; // r9
  __int64 v71; // rdx
  __int64 v72; // rcx
  __int64 v73; // r8
  __int64 v74; // rax
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  __int64 v78; // rcx
  __int64 v79; // r8
  __int64 v80; // r9
  __int64 v81; // rax
  unsigned int v82; // edi
  __int64 v83; // rax
  char i; // dl
  __int64 v85; // rdx
  __int64 v86; // [rsp-10h] [rbp-150h]
  __int64 v87; // [rsp-8h] [rbp-148h]
  unsigned __int64 *v88; // [rsp+0h] [rbp-140h]
  int *v89; // [rsp+8h] [rbp-138h]
  unsigned __int64 *v90; // [rsp+8h] [rbp-138h]
  __int64 v91; // [rsp+8h] [rbp-138h]
  __int64 v92; // [rsp+10h] [rbp-130h]
  __int64 *v93; // [rsp+10h] [rbp-130h]
  int *v94; // [rsp+10h] [rbp-130h]
  __int64 v95; // [rsp+10h] [rbp-130h]
  __int64 v96; // [rsp+18h] [rbp-128h]
  char *v97; // [rsp+18h] [rbp-128h]
  unsigned __int64 *v98; // [rsp+18h] [rbp-128h]
  __int64 v99; // [rsp+18h] [rbp-128h]
  __int64 v100; // [rsp+18h] [rbp-128h]
  int v101; // [rsp+18h] [rbp-128h]
  __int64 v102; // [rsp+18h] [rbp-128h]
  char v103; // [rsp+20h] [rbp-120h]
  unsigned int v104; // [rsp+20h] [rbp-120h]
  char v105; // [rsp+20h] [rbp-120h]
  __int64 v106; // [rsp+28h] [rbp-118h]
  unsigned __int64 v107; // [rsp+28h] [rbp-118h]
  __int64 v108; // [rsp+28h] [rbp-118h]
  __int64 v109; // [rsp+28h] [rbp-118h]
  __int64 v110; // [rsp+30h] [rbp-110h]
  __int64 v111; // [rsp+30h] [rbp-110h]
  __int64 v112; // [rsp+30h] [rbp-110h]
  int *v113; // [rsp+38h] [rbp-108h]
  int v114; // [rsp+44h] [rbp-FCh]
  __int64 *v115; // [rsp+48h] [rbp-F8h]
  __int64 v116; // [rsp+48h] [rbp-F8h]
  int *v117; // [rsp+50h] [rbp-F0h]
  __m128i *v118; // [rsp+58h] [rbp-E8h]
  unsigned int v119; // [rsp+58h] [rbp-E8h]
  int v120; // [rsp+64h] [rbp-DCh] BYREF
  unsigned int v121; // [rsp+68h] [rbp-D8h] BYREF
  unsigned int v122; // [rsp+6Ch] [rbp-D4h] BYREF
  __int64 v123; // [rsp+70h] [rbp-D0h] BYREF
  __int64 v124; // [rsp+78h] [rbp-C8h] BYREF
  __m128i v125; // [rsp+80h] [rbp-C0h] BYREF
  __m128i *v126; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v127; // [rsp+98h] [rbp-A8h]
  __int64 v128; // [rsp+A0h] [rbp-A0h]
  __int64 v129; // [rsp+A8h] [rbp-98h]
  __int64 v130[10]; // [rsp+B0h] [rbp-90h] BYREF
  int v131; // [rsp+104h] [rbp-3Ch]

  v7 = a1;
  v8 = &v120;
  v9 = (unsigned __int64 **)a2;
  v118 = a3;
  if ( a6 )
    v8 = a6;
  v114 = a4;
  v115 = a5;
  v117 = v8;
  v10 = *(_BYTE *)(a1 + 24);
  v113 = a6;
  v120 = 0;
  v121 = 0;
  if ( v10 == 32 )
  {
    v37 = *(_QWORD *)(a1 + 64);
    v38 = **(_QWORD **)(a1 + 56);
    v106 = *(_QWORD *)(a1 + 56);
    v110 = **(_QWORD **)(*(_QWORD *)(v38 + 88) + 32LL);
    if ( *(_QWORD *)(a2 + 16) )
    {
      v39 = qword_4F04C68[0] + 776LL * dword_4F04C64;
      v40 = *(_BYTE *)(v39 + 12) >> 5;
      *(_BYTE *)(v39 + 12) |= 0x20u;
      v41 = v40 & 1;
      if ( !a5 )
      {
        v102 = v38;
        v105 = v41;
        v115 = v130;
        sub_892150(v130);
        v41 = v105;
        v38 = v102;
        if ( (v114 & 0x40000) != 0 )
          v131 = 1;
      }
      v86 = (__int64)v115;
      v103 = v41;
      v116 = v38;
      v37 = sub_8A6360(v38, v37, v110, 0, a2, (int)a1 + 28, v114, (__int64)&v121, v86);
      *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) = *(_BYTE *)(qword_4F04C68[0]
                                                                           + 776LL * dword_4F04C64
                                                                           + 12)
                                                                & 0xDF
                                                                | (32 * (v103 & 1));
      v38 = v116;
    }
    sub_865900(v38);
    *(_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64 + 12) |= 0x20u;
    if ( v121 )
    {
      sub_686D90(0xBE7u, (FILE *)(a1 + 28), *(_QWORD *)(*(_QWORD *)a2 + 8LL), v118->m128i_i64);
      sub_864110(3047, a1 + 28, v42);
      if ( !v37 )
        goto LABEL_52;
    }
    else
    {
      v44 = v118->m128i_i64[1];
      v45 = *(_QWORD *)(v106 + 192);
      v46 = sub_6F1C10(v45, v37, v110, (_DWORD)v118, v114, 0, (__int64)v117, 0);
      if ( v46 )
      {
        v119 = v46;
        sub_864110(v45, v37, v87);
        result = v119;
        if ( v37 )
        {
          sub_725130(v37);
          result = v119;
        }
        goto LABEL_40;
      }
      if ( !*v117 )
      {
        v130[0] = 0;
        v130[1] = 0;
        sub_686D90(0xBE8u, (FILE *)(v7 + 28), v37, v130);
        sub_67E390(v130, v118->m128i_i64, v44);
        sub_864110(v130, v118, v85);
        goto LABEL_52;
      }
      sub_864110(v45, v37, v87);
      if ( !v37 )
      {
LABEL_52:
        if ( !*v117 )
          goto LABEL_39;
        goto LABEL_53;
      }
    }
    sub_725130(v37);
    goto LABEL_52;
  }
  if ( v10 != 1 )
    goto LABEL_5;
  v17 = *(_BYTE *)(a1 + 56);
  if ( v17 == 87 )
  {
    v18 = *(_QWORD *)(a1 + 72);
    if ( !(unsigned int)sub_6F0CB0(v18, a2, (_DWORD)a3, a4, (_DWORD)a5, (_DWORD)v117, (__int64)&v121) )
      goto LABEL_52;
    goto LABEL_19;
  }
  if ( v17 == 88 )
  {
    v18 = *(_QWORD *)(a1 + 72);
    if ( (unsigned int)sub_6F0CB0(v18, a2, (_DWORD)a3, a4, (_DWORD)a5, (_DWORD)v117, (__int64)&v121) )
      goto LABEL_14;
    if ( *v117 )
      goto LABEL_52;
LABEL_19:
    if ( (unsigned int)sub_6F0CB0(
                         *(_QWORD *)(v18 + 16),
                         a2,
                         (_DWORD)v118,
                         v114,
                         (_DWORD)v115,
                         (_DWORD)v117,
                         (__int64)&v121) )
      goto LABEL_14;
    goto LABEL_52;
  }
LABEL_5:
  v11 = *(_QWORD *)(a2 + 16);
  v12 = (_QWORD *)a1;
  v123 = 0;
  if ( v11 > 1 )
  {
    v19 = sub_724DC0(a1, a2, a3, a4, a5, a6);
    v126 = 0;
    v125.m128i_i64[0] = v19;
    v20 = *(_QWORD *)(a2 + 16);
    v127 = 0;
    v128 = 0;
    v21 = v20 - 1;
    v126 = (__m128i *)sub_823970(24 * (v20 - 1));
    v22 = v126;
    v127 = v21;
    v23 = v115;
    if ( !v115 )
    {
      sub_892150(v130);
      v21 = v127;
      v22 = v126;
      v23 = v130;
    }
    v24 = v9[2];
    v25 = v128;
    v26 = *v9;
    v27 = (__int64)v24 - 1;
    v28 = (__int64)v24 + v128 - 1;
    v29 = (const __m128i *)(*v9 + 3);
    if ( v21 < v28 )
    {
      v88 = v9[2];
      v90 = *v9;
      v93 = v23;
      v97 = (char *)v24 - 1;
      v108 = (__int64)v24 + v128 - 1;
      v57 = (__m128i *)sub_823970(24 * v28);
      v112 = (__int64)v57;
      v58 = (__int64)v97;
      v59 = v90;
      if ( v25 > 0 )
      {
        v60 = v22;
        v61 = (__m128i *)((char *)v57 + 24 * v25);
        do
        {
          if ( v57 )
          {
            *v57 = _mm_loadu_si128(v60);
            v57[1].m128i_i64[0] = v60[1].m128i_i64[0];
          }
          v57 = (__m128i *)((char *)v57 + 24);
          v60 = (const __m128i *)((char *)v60 + 24);
        }
        while ( v61 != v57 );
        v58 = (__int64)v97;
      }
      v91 = v108;
      v98 = v59;
      v109 = v58;
      sub_823A00(v22, 24 * v21);
      v22 = (__m128i *)v112;
      v27 = v109;
      v23 = v93;
      v126 = (__m128i *)v112;
      v26 = v98;
      v127 = v91;
      v24 = v88;
    }
    if ( v25 > 0 )
    {
      v30 = (__m128i *)((char *)v22 + 24 * v25 - 24);
      v31 = (__m128i *)((char *)v22 + 24 * ((_QWORD)v24 + v25) - 48);
      while ( 1 )
      {
        if ( v31 )
        {
          *v31 = _mm_loadu_si128(v30);
          v31[1].m128i_i64[0] = v30[1].m128i_i64[0];
        }
        v31 = (__m128i *)((char *)v31 - 24);
        if ( v22 == v30 )
          break;
        v30 = (const __m128i *)((char *)v30 - 24);
      }
    }
    if ( v27 > 0 )
    {
      do
      {
        if ( v22 )
        {
          *v22 = _mm_loadu_si128(v29);
          v22[1].m128i_i64[0] = v29[1].m128i_i64[0];
        }
        v29 = (const __m128i *)((char *)v29 + 24);
        v22 = (__m128i *)((char *)v22 + 24);
      }
      while ( v29 != (const __m128i *)&v26[3 * (_QWORD)v24] );
    }
    v128 += v27;
    v32 = sub_6EFFF0(v7, &v126, (__int64)v23, 0x4000, (const __m128i *)v125.m128i_i64[0], &v123, &v121);
    v12 = (_QWORD *)v32;
    if ( v121 || v32 )
    {
      sub_724E30(&v125);
    }
    else
    {
      if ( v123 )
        sub_724E30(&v125);
      else
        v123 = sub_724E50(&v125, &v126, v33, v34, v35);
      v12 = (_QWORD *)sub_73A720(v123);
    }
    a1 = (__int64)v126;
    v123 = 0;
    a2 = 24 * v127;
    sub_823A00(v126, 24 * v127);
    if ( v121 )
      goto LABEL_38;
    v11 = (__int64)v9[2];
  }
  if ( !v11 )
  {
LABEL_7:
    if ( v12 )
      goto LABEL_8;
LABEL_99:
    v64 = v123;
LABEL_100:
    if ( (unsigned int)sub_8D29A0(*(_QWORD *)(v64 + 128)) )
    {
      if ( *(_BYTE *)(v123 + 173) != 12 && !(unsigned int)sub_711520(v123, a2) )
        goto LABEL_14;
      sub_6855B0(0xBE3u, (FILE *)(v7 + 28), v118);
    }
    else
    {
      v83 = *(_QWORD *)(v123 + 128);
      for ( i = *(_BYTE *)(v83 + 140); i == 12; i = *(_BYTE *)(v83 + 140) )
        v83 = *(_QWORD *)(v83 + 160);
      *v117 = i != 0;
      sub_6855B0(0xBE0u, (FILE *)(v7 + 28), v118);
    }
    goto LABEL_52;
  }
  v47 = (*v9)[1];
  a2 = **v9;
  v107 = v47;
  v111 = a2;
  v48 = 31 * (31 * (31 * ((v7 >> 3) + 527) + (a2 >> 3)) + (unsigned int)sub_72A8B0(v12));
  v104 = sub_72E120(v47) + v48;
  a5 = (__int64 *)*(unsigned int *)(qword_4D03A48 + 8);
  v49 = v12;
  v50 = (unsigned int)a5 & v104;
  v51 = *(_QWORD *)qword_4D03A48;
  v52 = *(_DWORD *)(qword_4D03A48 + 8);
  a4 = (__int64)v49;
  while ( 1 )
  {
    v53 = (int **)(v51 + 48LL * v50);
    a6 = *v53;
    a1 = (__int64)v53[1];
    v54 = v53[2];
    v55 = (__int64)v53[3];
    if ( (int *)v7 != *v53 )
      goto LABEL_60;
    if ( (int *)v111 != v54 )
      goto LABEL_61;
    if ( a1 == a4 )
      goto LABEL_133;
    a2 = a4;
    v89 = *v53;
    v92 = (__int64)v53[3];
    v96 = a4;
    v56 = sub_7386E0(a1, a4, 7, a4, a5);
    a4 = v96;
    v55 = v92;
    a6 = v89;
    if ( v56 )
    {
LABEL_133:
      if ( v55 == v107 )
        break;
      if ( v55 )
      {
        if ( v107 )
        {
          a2 = v107;
          a1 = v55;
          v94 = a6;
          v99 = a4;
          v62 = sub_89AB40(v55, v107, 80);
          a4 = v99;
          a6 = v94;
          if ( v62 )
            break;
        }
      }
    }
    v55 = (__int64)v53[3];
    a1 = (__int64)v53[1];
    v54 = v53[2];
    a6 = *v53;
LABEL_60:
    if ( !a6 && !v54 )
    {
      if ( !a1 || (a2 = 0, v95 = a4, v100 = v55, v65 = sub_7386E0(a1, 0, 7, a4, a5), v55 = v100, a4 = v95, v65) )
      {
        if ( !v55 )
        {
          v125 = 0u;
          v63 = a4;
          goto LABEL_90;
        }
      }
    }
LABEL_61:
    v50 = v52 & (v50 + 1);
  }
  a2 = (unsigned __int64)v53[4];
  v63 = a4;
  a4 = (__int64)v53[5];
  v125.m128i_i64[0] = a2;
  a3 = (__m128i *)(unsigned int)a2;
  v125.m128i_i64[1] = a4;
  v64 = a4;
  switch ( (_DWORD)a2 )
  {
    case 0:
LABEL_90:
      v66 = dword_4F07508[1];
      v101 = dword_4F07508[0];
      v124 = sub_724DC0(a1, a2, a3, a4, a5, a6);
      sub_7296C0(&v122);
      v67 = sub_72F240(v107);
      v126 = (__m128i *)v7;
      v127 = v63;
      v128 = v111;
      v129 = v67;
      v125.m128i_i32[0] = 1;
      sub_6F0A70((__int64 *)qword_4D03A48, &v125, v104, v68, v69, v70, v7, v63, v111, v67);
      if ( !v115 )
      {
        v115 = v130;
        sub_892150(v130);
        if ( (v114 & 0x40000) != 0 )
          v131 = 1;
      }
      v12 = (_QWORD *)sub_7410C0(
                        v63,
                        v107,
                        v111,
                        0,
                        (int)v63 + 28,
                        v114,
                        (__int64)&v121,
                        (__int64)v115,
                        v124,
                        (__int64)&v123);
      if ( v12 )
      {
        v74 = 0;
        v125.m128i_i32[0] = 2;
        if ( !v121 )
          v74 = (__int64)v12;
      }
      else
      {
        if ( !v121 )
        {
          if ( v123 )
          {
            sub_724E30(&v124);
            v81 = v123;
          }
          else
          {
            v81 = sub_724E50(&v124, v107, v71, v72, v73);
            v123 = v81;
          }
          a2 = (unsigned __int64)&v125;
          v125.m128i_i64[1] = v81;
          v125.m128i_i32[0] = 3;
          sub_6F0A70((__int64 *)qword_4D03A48, &v125, v104, v78, v79, v80, (__int64)v126, v127, v128, v129);
          sub_729730(v122);
          v82 = v121;
          LOWORD(dword_4F07508[1]) = v66;
          dword_4F07508[0] = v101;
          if ( v82 )
            break;
          goto LABEL_99;
        }
        v125.m128i_i32[0] = 2;
        v74 = 0;
      }
      v125.m128i_i64[1] = v74;
      sub_724E30(&v124);
      a2 = (unsigned __int64)&v125;
      sub_6F0A70((__int64 *)qword_4D03A48, &v125, v104, v75, v76, v77, (__int64)v126, v127, v128, v129);
      a1 = v122;
      sub_729730(v122);
      LOWORD(dword_4F07508[1]) = v66;
      a5 = (__int64 *)v121;
      dword_4F07508[0] = v101;
      if ( !v121 )
        goto LABEL_7;
      break;
    case 2:
      if ( !a4 )
      {
        v121 = 1;
        v123 = 0;
        break;
      }
      a2 = v121;
      v123 = 0;
      if ( v121 )
        break;
      v12 = (_QWORD *)a4;
LABEL_8:
      v130[0] = sub_724DC0(a1, a2, a3, a4, a5, a6);
      v13 = *(_QWORD *)sub_6E4240((__int64)v12, 0);
      if ( (unsigned int)sub_8D29A0(v13) )
      {
        if ( (*((_BYTE *)v12 + 25) & 3) != 0 )
          v12 = sub_6ED3D0((__int64)v12, 0, 0, 0, v14, v15);
        if ( (unsigned int)sub_7A30C0(v12, 1, 1, v130[0]) )
        {
          if ( !(unsigned int)sub_711520(v130[0], 1) )
          {
            sub_724E30(v130);
LABEL_14:
            result = 1;
            goto LABEL_40;
          }
          sub_6855B0(0xBE3u, (FILE *)(v7 + 28), v118);
          sub_724E30(v130);
        }
        else
        {
          *v117 = 1;
          sub_6855B0(0xBE2u, (FILE *)((char *)v12 + 28), v118);
          sub_724E30(v130);
        }
      }
      else
      {
        while ( 1 )
        {
          v36 = *(_BYTE *)(v13 + 140);
          if ( v36 != 12 )
            break;
          v13 = *(_QWORD *)(v13 + 160);
        }
        *v117 = v36 != 0;
        sub_6855B0(0xBE0u, (FILE *)(v7 + 28), v118);
        sub_724E30(v130);
      }
      goto LABEL_52;
    case 3:
      v123 = a4;
      if ( !v121 )
        goto LABEL_100;
      break;
    case 1:
      sub_6851C0(0xCC6u, a6 + 7);
      v121 = 1;
      *v117 = 1;
      break;
    default:
      sub_721090(a1);
  }
LABEL_38:
  sub_6855B0(0xBE1u, (FILE *)(v7 + 28), v118);
  if ( !*v117 )
  {
LABEL_39:
    result = 0;
    goto LABEL_40;
  }
LABEL_53:
  if ( v113 )
    goto LABEL_39;
  v43 = sub_67D9D0(0xBE4u, dword_4F07508);
  sub_67E370((__int64)v43, v118);
  sub_685910((__int64)v43, (FILE *)v118);
  result = 0;
LABEL_40:
  if ( a7 )
    *a7 = v121;
  return result;
}
