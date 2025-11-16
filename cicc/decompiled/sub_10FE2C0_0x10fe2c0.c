// Function: sub_10FE2C0
// Address: 0x10fe2c0
//
char __fastcall sub_10FE2C0(unsigned __int8 *a1, __int64 a2, const __m128i *a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r14
  __int64 v7; // r15
  int v8; // eax
  __int64 v11; // rdx
  char result; // al
  __int64 v13; // rdi
  unsigned __int8 *v15; // rdi
  __int64 v16; // r8
  __int64 v17; // r9
  unsigned __int8 *v18; // r12
  __int64 v19; // rdi
  __int64 v20; // rax
  unsigned __int8 *v21; // rbx
  __int64 v22; // r12
  unsigned int v23; // r15d
  unsigned int v24; // ebx
  __int64 *v25; // rdx
  __int64 v26; // rdi
  __m128i v27; // xmm0
  __m128i v28; // xmm1
  __m128i v29; // xmm3
  __int64 v30; // rax
  unsigned __int8 *v31; // rcx
  __int64 v32; // rdi
  __m128i v33; // xmm5
  __int64 v34; // rax
  unsigned __int64 v35; // xmm6_8
  __m128i v36; // xmm7
  __int64 v37; // r8
  __int64 v38; // r9
  __int64 v39; // rdi
  __int64 v40; // rdx
  __int64 v41; // rdi
  unsigned int *v42; // rax
  unsigned int v43; // ebx
  unsigned __int64 v44; // rbx
  unsigned __int8 *v45; // rcx
  unsigned __int64 v46; // r8
  __int64 v47; // r9
  __int32 v48; // edx
  __int64 v49; // rax
  unsigned __int64 v50; // rax
  unsigned __int8 *v51; // rdi
  __int64 v52; // r8
  __int64 v53; // r9
  __int64 v54; // rdi
  unsigned int v55; // ebx
  unsigned __int8 *v56; // rcx
  __int32 v57; // r10d
  __int64 v58; // r11
  unsigned __int64 v59; // r11
  __int64 *v60; // rdi
  unsigned int v61; // eax
  unsigned int v62; // ebx
  unsigned int v63; // eax
  unsigned __int8 *v64; // rcx
  unsigned int v65; // r11d
  __int32 v66; // r8d
  __int64 v67; // r9
  unsigned __int64 v68; // r9
  __int64 *v69; // rcx
  __int64 v70; // rdi
  __m128i v71; // xmm4
  __m128i v72; // xmm5
  unsigned __int64 v73; // xmm6_8
  __m128i v74; // xmm7
  char v75; // al
  __int64 v76; // r8
  int v77; // eax
  int v78; // eax
  int v79; // eax
  unsigned __int8 *v80; // r12
  unsigned __int8 *v81; // rdi
  unsigned __int8 *v82; // rdi
  __int64 v83; // r8
  __int64 v84; // r9
  unsigned __int8 *v85; // r12
  __int64 v86; // rax
  __int64 v87; // r8
  __int64 v88; // r9
  int v89; // [rsp-E0h] [rbp-E0h]
  unsigned __int64 *v90; // [rsp-E0h] [rbp-E0h]
  unsigned __int64 v91; // [rsp-D8h] [rbp-D8h]
  unsigned int v92; // [rsp-D8h] [rbp-D8h]
  unsigned __int64 v93; // [rsp-D8h] [rbp-D8h]
  __int32 v94; // [rsp-D8h] [rbp-D8h]
  _QWORD *v95; // [rsp-D8h] [rbp-D8h]
  unsigned int v96; // [rsp-D8h] [rbp-D8h]
  unsigned __int8 *v97; // [rsp-D0h] [rbp-D0h]
  char v98; // [rsp-D0h] [rbp-D0h]
  char v99; // [rsp-D0h] [rbp-D0h]
  unsigned int v100; // [rsp-D0h] [rbp-D0h]
  unsigned __int64 v101; // [rsp-D0h] [rbp-D0h]
  unsigned int v102; // [rsp-D0h] [rbp-D0h]
  unsigned int v103; // [rsp-D0h] [rbp-D0h]
  unsigned __int64 *v104; // [rsp-D0h] [rbp-D0h]
  unsigned int v105; // [rsp-D0h] [rbp-D0h]
  char v106; // [rsp-D0h] [rbp-D0h]
  char v107; // [rsp-D0h] [rbp-D0h]
  __int64 v108; // [rsp-C8h] [rbp-C8h] BYREF
  unsigned int v109; // [rsp-C0h] [rbp-C0h]
  __int64 v110; // [rsp-B8h] [rbp-B8h] BYREF
  __int32 v111; // [rsp-B0h] [rbp-B0h]
  unsigned __int64 v112; // [rsp-A8h] [rbp-A8h] BYREF
  unsigned int v113; // [rsp-A0h] [rbp-A0h]
  __int64 v114; // [rsp-98h] [rbp-98h]
  unsigned int v115; // [rsp-90h] [rbp-90h]
  __m128i v116; // [rsp-88h] [rbp-88h] BYREF
  __m128i v117; // [rsp-78h] [rbp-78h]
  unsigned __int64 v118; // [rsp-68h] [rbp-68h]
  unsigned __int8 *v119; // [rsp-60h] [rbp-60h]
  __m128i v120; // [rsp-58h] [rbp-58h]
  __int64 v121; // [rsp-48h] [rbp-48h]
  __int64 v122; // [rsp-18h] [rbp-18h]
  __int64 v123; // [rsp-10h] [rbp-10h]

  v8 = *a1;
  if ( (unsigned __int8)v8 <= 0x1Cu )
    return 0;
  v123 = v7;
  v122 = v6;
  v11 = *((_QWORD *)a1 + 2);
  if ( !v11 || *(_QWORD *)(v11 + 8) )
    return 0;
  v13 = *((_QWORD *)a1 + 1);
  switch ( v8 )
  {
    case '*':
    case ',':
    case '.':
    case '9':
    case ':':
    case ';':
    case '\\':
      if ( (a1[7] & 0x40) != 0 )
        v15 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        v15 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      if ( !(unsigned __int8)sub_10FF0A0(*(_QWORD *)v15, a2, a3, a4, a5, a6) )
        return 0;
      if ( (a1[7] & 0x40) != 0 )
        v18 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        v18 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v19 = *((_QWORD *)v18 + 4);
      return sub_10FF0A0(v19, a2, a3, a4, v16, v17);
    case '0':
    case '3':
      v23 = sub_BCB060(v13);
      v113 = v23;
      v24 = sub_BCB060(a2);
      if ( v23 > 0x40 )
      {
        sub_C43690((__int64)&v112, 0, 0);
        v23 = v113;
      }
      else
      {
        v112 = 0;
      }
      if ( v24 != v23 )
      {
        if ( v24 > 0x3F || v23 > 0x40 )
          sub_C43C90(&v112, v24, v23);
        else
          v112 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v24 + 64 - (unsigned __int8)v23) << v24;
      }
      if ( (a1[7] & 0x40) != 0 )
        v25 = (__int64 *)*((_QWORD *)a1 - 1);
      else
        v25 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v26 = *v25;
      v27 = _mm_loadu_si128(a3 + 6);
      v28 = _mm_loadu_si128(a3 + 7);
      v29 = _mm_loadu_si128(a3 + 9);
      v30 = a3[10].m128i_i64[0];
      v118 = _mm_loadu_si128(a3 + 8).m128i_u64[0];
      v121 = v30;
      v119 = a1;
      v116 = v27;
      v117 = v28;
      v120 = v29;
      if ( !(unsigned __int8)sub_9AC230(v26, (__int64)&v112, &v116, 0) )
        goto LABEL_35;
      v31 = (a1[7] & 0x40) != 0 ? (unsigned __int8 *)*((_QWORD *)a1 - 1) : &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v32 = *((_QWORD *)v31 + 4);
      v33 = _mm_loadu_si128(a3 + 7);
      v34 = a3[10].m128i_i64[0];
      v35 = _mm_loadu_si128(a3 + 8).m128i_u64[0];
      v116 = _mm_loadu_si128(a3 + 6);
      v36 = _mm_loadu_si128(a3 + 9);
      v121 = v34;
      v118 = v35;
      v117 = v33;
      v119 = a1;
      v120 = v36;
      if ( !(unsigned __int8)sub_9AC230(v32, (__int64)&v112, &v116, 0) )
        goto LABEL_35;
      if ( (a1[7] & 0x40) != 0 )
        v81 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        v81 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      result = sub_10FF0A0(*(_QWORD *)v81, a2, a3, a1, v37, v38);
      if ( result )
      {
        v86 = sub_986520((__int64)a1);
        result = sub_10FF0A0(*(_QWORD *)(v86 + 32), a2, a3, a1, v87, v88);
      }
      goto LABEL_146;
    case '6':
      v44 = (unsigned int)sub_BCB060(a2);
      if ( (a1[7] & 0x40) != 0 )
        v45 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        v45 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      sub_9AC3E0((__int64)&v116, *((_QWORD *)v45 + 4), a3[5].m128i_u64[1], 0, 0, 0, 0, 1);
      v48 = v116.m128i_i32[2];
      v113 = v116.m128i_u32[2];
      if ( v116.m128i_i32[2] <= 0x40u )
      {
        v49 = v116.m128i_i64[0];
        goto LABEL_49;
      }
      sub_C43780((__int64)&v112, (const void **)&v116);
      v48 = v113;
      if ( v113 <= 0x40 )
      {
        v49 = v112;
LABEL_49:
        v111 = v48;
        v50 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v48) & ~v49;
        if ( !v48 )
          v50 = 0;
        v110 = v50;
        if ( v44 <= v50 )
          goto LABEL_73;
        goto LABEL_52;
      }
      sub_C43D10((__int64)&v112);
      v46 = v112;
      v111 = v113;
      v110 = v112;
      v96 = v113;
      if ( v113 <= 0x40 )
      {
        if ( v112 < v44 )
          goto LABEL_52;
      }
      else
      {
        v104 = (unsigned __int64 *)v112;
        v79 = sub_C444A0((__int64)&v110);
        v46 = (unsigned __int64)v104;
        if ( v96 - v79 <= 0x40 && *v104 < v44 )
        {
          if ( v104 )
            j_j___libc_free_0_0(v104);
          goto LABEL_52;
        }
        if ( v104 )
          j_j___libc_free_0_0(v104);
      }
LABEL_73:
      if ( v117.m128i_i32[2] > 0x40u && v117.m128i_i64[0] )
        j_j___libc_free_0_0(v117.m128i_i64[0]);
      if ( v116.m128i_i32[2] > 0x40u )
      {
        v39 = v116.m128i_i64[0];
        if ( v116.m128i_i64[0] )
LABEL_37:
          j_j___libc_free_0_0(v39);
      }
      return 0;
    case '7':
      v62 = sub_BCB060(v13);
      v63 = sub_BCB060(a2);
      if ( (a1[7] & 0x40) != 0 )
        v64 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        v64 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v92 = v63;
      sub_9AC3E0((__int64)&v112, *((_QWORD *)v64 + 4), a3[5].m128i_u64[1], 0, 0, 0, 0, 1);
      v109 = v62;
      v65 = v92;
      if ( v62 > 0x40 )
      {
        sub_C43690((__int64)&v108, 0, 0);
        v62 = v109;
        v65 = v92;
      }
      else
      {
        v108 = 0;
      }
      if ( v65 != v62 )
      {
        if ( v65 > 0x3F || v62 > 0x40 )
        {
          v105 = v65;
          sub_C43C90(&v108, v65, v62);
          v65 = v105;
        }
        else
        {
          v108 |= 0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v65 + 64 - (unsigned __int8)v62) << v65;
        }
      }
      v66 = v113;
      v101 = v65;
      v116.m128i_i32[2] = v113;
      if ( v113 <= 0x40 )
      {
        v67 = v112;
LABEL_89:
        v111 = v66;
        v68 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v66) & ~v67;
        if ( !v66 )
          v68 = 0;
        v110 = v68;
LABEL_92:
        if ( v101 <= v68 )
          goto LABEL_97;
        goto LABEL_93;
      }
      sub_C43780((__int64)&v116, (const void **)&v112);
      v66 = v116.m128i_i32[2];
      if ( v116.m128i_i32[2] <= 0x40u )
      {
        v67 = v116.m128i_i64[0];
        goto LABEL_89;
      }
      sub_C43D10((__int64)&v116);
      v66 = v116.m128i_i32[2];
      v68 = v116.m128i_i64[0];
      v111 = v116.m128i_i32[2];
      v110 = v116.m128i_i64[0];
      if ( v116.m128i_i32[2] <= 0x40u )
        goto LABEL_92;
      v90 = (unsigned __int64 *)v116.m128i_i64[0];
      v94 = v116.m128i_i32[2];
      v77 = sub_C444A0((__int64)&v110);
      v66 = v94;
      v68 = (unsigned __int64)v90;
      if ( (unsigned int)(v94 - v77) > 0x40 || *v90 >= v101 )
        goto LABEL_106;
LABEL_93:
      if ( (a1[7] & 0x40) != 0 )
        v69 = (__int64 *)*((_QWORD *)a1 - 1);
      else
        v69 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v70 = *v69;
      v71 = _mm_loadu_si128(a3 + 6);
      v72 = _mm_loadu_si128(a3 + 7);
      v93 = v68;
      v73 = _mm_loadu_si128(a3 + 8).m128i_u64[0];
      v102 = v66;
      v74 = _mm_loadu_si128(a3 + 9);
      v121 = a3[10].m128i_i64[0];
      v118 = v73;
      v116 = v71;
      v119 = (unsigned __int8 *)a4;
      v117 = v72;
      v120 = v74;
      v75 = sub_9AC230(v70, (__int64)&v108, &v116, 0);
      v76 = v102;
      v68 = v93;
      if ( !v75 )
      {
        if ( v102 <= 0x40 )
        {
LABEL_97:
          if ( v109 > 0x40 && v108 )
            j_j___libc_free_0_0(v108);
          if ( v115 > 0x40 && v114 )
            j_j___libc_free_0_0(v114);
LABEL_35:
          if ( v113 <= 0x40 )
            return 0;
          v39 = v112;
          if ( !v112 )
            return 0;
          goto LABEL_37;
        }
LABEL_106:
        if ( v68 )
          j_j___libc_free_0_0(v68);
        goto LABEL_97;
      }
      if ( v102 > 0x40 && v93 )
        j_j___libc_free_0_0(v93);
      if ( (a1[7] & 0x40) != 0 )
        v82 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        v82 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      result = sub_10FF0A0(*(_QWORD *)v82, a2, a3, a4, v76, v68);
      if ( result )
      {
        if ( (a1[7] & 0x40) != 0 )
          v85 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
        else
          v85 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
        result = sub_10FF0A0(*((_QWORD *)v85 + 4), a2, a3, a4, v83, v84);
      }
      if ( v109 > 0x40 && v108 )
      {
        v106 = result;
        j_j___libc_free_0_0(v108);
        result = v106;
      }
      if ( v115 > 0x40 && v114 )
      {
        v107 = result;
        j_j___libc_free_0_0(v114);
        result = v107;
      }
LABEL_146:
      if ( v113 > 0x40 )
      {
        v54 = v112;
        if ( v112 )
          goto LABEL_60;
      }
      return result;
    case '8':
      v89 = sub_BCB060(v13);
      v55 = sub_BCB060(a2);
      if ( (a1[7] & 0x40) != 0 )
        v56 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        v56 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      sub_9AC3E0((__int64)&v116, *((_QWORD *)v56 + 4), a3[5].m128i_u64[1], 0, 0, 0, 0, 1);
      v57 = v116.m128i_i32[2];
      v113 = v116.m128i_u32[2];
      if ( v116.m128i_i32[2] <= 0x40u )
      {
        v58 = v116.m128i_i64[0];
LABEL_65:
        v111 = v57;
        v59 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v57) & ~v58;
        if ( !v57 )
          v59 = 0;
        v110 = v59;
LABEL_68:
        if ( v59 >= v55 )
          goto LABEL_73;
        goto LABEL_69;
      }
      sub_C43780((__int64)&v112, (const void **)&v116);
      v57 = v113;
      if ( v113 <= 0x40 )
      {
        v58 = v112;
        goto LABEL_65;
      }
      sub_C43D10((__int64)&v112);
      v57 = v113;
      v59 = v112;
      v111 = v113;
      v110 = v112;
      if ( v113 <= 0x40 )
        goto LABEL_68;
      v95 = (_QWORD *)v112;
      v103 = v113;
      v78 = sub_C444A0((__int64)&v110);
      v57 = v103;
      v59 = (unsigned __int64)v95;
      if ( v103 - v78 > 0x40 || *v95 >= (unsigned __int64)v55 )
      {
LABEL_115:
        if ( v59 )
          j_j___libc_free_0_0(v59);
        goto LABEL_73;
      }
LABEL_69:
      if ( (a1[7] & 0x40) != 0 )
        v60 = (__int64 *)*((_QWORD *)a1 - 1);
      else
        v60 = (__int64 *)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v91 = v59;
      v100 = v57;
      v61 = sub_9AF8B0(*v60, a3[5].m128i_u64[1], 0, a3[4].m128i_i64[0], a4, a3[5].m128i_i64[0], 1);
      v59 = v91;
      if ( v89 - v55 >= v61 )
      {
        if ( v100 <= 0x40 )
          goto LABEL_73;
        goto LABEL_115;
      }
      if ( v100 > 0x40 && v91 )
        j_j___libc_free_0_0(v91);
LABEL_52:
      if ( (a1[7] & 0x40) != 0 )
        v51 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
      else
        v51 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      result = sub_10FF0A0(*(_QWORD *)v51, a2, a3, a4, v46, v47);
      if ( result )
      {
        if ( (a1[7] & 0x40) != 0 )
          v80 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
        else
          v80 = &a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
        result = sub_10FF0A0(*((_QWORD *)v80 + 4), a2, a3, a4, v52, v53);
      }
      if ( v117.m128i_i32[2] > 0x40u && v117.m128i_i64[0] )
      {
        v98 = result;
        j_j___libc_free_0_0(v117.m128i_i64[0]);
        result = v98;
      }
      if ( v116.m128i_i32[2] > 0x40u )
      {
        v54 = v116.m128i_i64[0];
        if ( v116.m128i_i64[0] )
        {
LABEL_60:
          v99 = result;
          j_j___libc_free_0_0(v54);
          result = v99;
        }
      }
      break;
    case 'C':
    case 'D':
    case 'E':
      return 1;
    case 'F':
    case 'G':
      if ( (a1[7] & 0x40) != 0 )
        v40 = *((_QWORD *)a1 - 1);
      else
        v40 = (__int64)&a1[-32 * (*((_DWORD *)a1 + 1) & 0x7FFFFFF)];
      v41 = *(_QWORD *)(*(_QWORD *)v40 + 8LL);
      if ( (unsigned int)*(unsigned __int8 *)(v41 + 8) - 17 <= 1 )
        v41 = **(_QWORD **)(v41 + 16);
      v42 = (unsigned int *)sub_BCAC60(v41, a2, v40, a4, a5);
      v43 = sub_C336E0(v42, *a1 == 71);
      return v43 <= (unsigned int)sub_BCB060(a2);
    case 'T':
      v20 = 32LL * (*((_DWORD *)a1 + 1) & 0x7FFFFFF);
      if ( (a1[7] & 0x40) != 0 )
      {
        v21 = (unsigned __int8 *)*((_QWORD *)a1 - 1);
        v97 = &v21[v20];
      }
      else
      {
        v97 = a1;
        v21 = &a1[-v20];
      }
      if ( v21 == v97 )
        return 1;
      while ( 1 )
      {
        v22 = *(_QWORD *)v21;
        if ( !(unsigned __int8)sub_10FD310(*(_BYTE **)v21, a2) && !(unsigned __int8)sub_10FE2C0(v22, a2, a3, a4) )
          break;
        v21 += 32;
        if ( v97 == v21 )
          return 1;
      }
      return 0;
    case 'V':
      if ( !(unsigned __int8)sub_10FF0A0(*((_QWORD *)a1 - 8), a2, a3, a4, a5, a6) )
        return 0;
      v19 = *((_QWORD *)a1 - 4);
      return sub_10FF0A0(v19, a2, a3, a4, v16, v17);
    default:
      return 0;
  }
  return result;
}
