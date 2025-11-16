// Function: sub_2744150
// Address: 0x2744150
//
__int64 __fastcall sub_2744150(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v8; // r15
  bool v9; // r8
  __int64 result; // rax
  __int64 v11; // r8
  __int64 v12; // rdi
  __int64 v13; // r15
  __int64 v14; // rax
  unsigned int v15; // eax
  __int64 v16; // r12
  unsigned int v17; // r14d
  int v18; // r14d
  int v19; // r13d
  __m128i *v20; // r8
  unsigned __int64 v21; // rsi
  __int64 v22; // rdx
  unsigned __int64 v23; // rdi
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // r9
  const __m128i *v26; // r15
  __int64 v27; // rdi
  unsigned __int64 v28; // r9
  __int64 v29; // rdx
  unsigned __int64 v30; // rsi
  unsigned __int64 v31; // rcx
  _BYTE *v32; // rax
  char v33; // al
  __int64 v34; // r10
  unsigned int v35; // r9d
  int v36; // r8d
  __int64 v37; // rax
  unsigned int v38; // r9d
  unsigned __int8 *v39; // rdx
  int v40; // r8d
  _BYTE *v41; // rcx
  unsigned int v42; // eax
  _BYTE *v43; // rax
  char v44; // al
  __int64 v45; // r10
  unsigned int v46; // r9d
  int v47; // r8d
  __int64 v48; // rax
  _BYTE *v49; // rax
  char v50; // al
  __int64 v51; // r10
  unsigned int v52; // r9d
  int v53; // r8d
  _BYTE *v54; // rax
  char v55; // al
  __int64 v56; // r10
  unsigned int v57; // r9d
  int v58; // r8d
  _BYTE *v59; // rax
  char v60; // al
  __int64 v61; // r10
  unsigned int v62; // r9d
  int v63; // r8d
  _BYTE *v64; // rax
  char v65; // al
  __int64 v66; // r10
  unsigned int v67; // r9d
  int v68; // r8d
  __int8 *v69; // r15
  __m128i *v70; // r15
  unsigned __int64 v71; // rax
  __m128i *v72; // rax
  _DWORD *v73; // rax
  int v74; // edx
  unsigned __int64 v75; // rax
  char v76; // al
  _DWORD *v77; // rax
  int v78; // edx
  char v79; // al
  _DWORD *v80; // rax
  __int64 v81; // r10
  int v82; // edx
  unsigned __int64 v83; // rax
  char v84; // al
  _DWORD *v85; // rax
  __int64 v86; // r10
  int v87; // edx
  unsigned __int64 v88; // rax
  char v89; // al
  _DWORD *v90; // rax
  __int64 v91; // r10
  int v92; // edx
  unsigned __int64 v93; // rax
  char v94; // al
  _BYTE *v95; // rax
  const void *v96; // rsi
  __int64 v97; // [rsp+0h] [rbp-B0h]
  __int64 v98; // [rsp+10h] [rbp-A0h]
  __int64 v99; // [rsp+18h] [rbp-98h]
  int v100; // [rsp+18h] [rbp-98h]
  unsigned int v101; // [rsp+18h] [rbp-98h]
  int v102; // [rsp+18h] [rbp-98h]
  __int64 v103; // [rsp+18h] [rbp-98h]
  int v104; // [rsp+18h] [rbp-98h]
  __int64 v105; // [rsp+18h] [rbp-98h]
  int v106; // [rsp+18h] [rbp-98h]
  __int64 v107; // [rsp+18h] [rbp-98h]
  int v108; // [rsp+18h] [rbp-98h]
  int v109; // [rsp+20h] [rbp-90h]
  unsigned int v110; // [rsp+20h] [rbp-90h]
  int v111; // [rsp+20h] [rbp-90h]
  unsigned int v112; // [rsp+20h] [rbp-90h]
  int v113; // [rsp+20h] [rbp-90h]
  unsigned int v114; // [rsp+20h] [rbp-90h]
  int v115; // [rsp+20h] [rbp-90h]
  unsigned int v116; // [rsp+20h] [rbp-90h]
  int v117; // [rsp+20h] [rbp-90h]
  unsigned int v118; // [rsp+20h] [rbp-90h]
  unsigned __int64 v119; // [rsp+20h] [rbp-90h]
  unsigned int v120; // [rsp+28h] [rbp-88h]
  __int64 v121; // [rsp+28h] [rbp-88h]
  unsigned int v122; // [rsp+28h] [rbp-88h]
  unsigned int v123; // [rsp+28h] [rbp-88h]
  __int64 v124; // [rsp+28h] [rbp-88h]
  __m128i *v125; // [rsp+28h] [rbp-88h]
  unsigned int v126; // [rsp+28h] [rbp-88h]
  __int64 v127; // [rsp+28h] [rbp-88h]
  unsigned int v128; // [rsp+28h] [rbp-88h]
  __int64 v129; // [rsp+28h] [rbp-88h]
  unsigned int v130; // [rsp+28h] [rbp-88h]
  __int64 v131; // [rsp+28h] [rbp-88h]
  __m128i v132; // [rsp+30h] [rbp-80h] BYREF
  __int64 v133; // [rsp+40h] [rbp-70h]
  __int64 v134; // [rsp+48h] [rbp-68h]
  __int64 v135; // [rsp+50h] [rbp-60h]
  __int64 v136; // [rsp+58h] [rbp-58h]
  __int64 v137; // [rsp+60h] [rbp-50h]
  __int64 v138; // [rsp+68h] [rbp-48h]
  __int16 v139; // [rsp+70h] [rbp-40h]

  v8 = *a1;
  v9 = sub_B532B0(a2);
  result = v8 + 632;
  if ( !v9 )
    result = v8;
  if ( *(_DWORD *)(result + 16) <= (unsigned int)qword_4FFA1C8 )
  {
    sub_27440D0(*a1, a2, (unsigned __int8 *)a3, (_BYTE *)a4, *(_DWORD *)(a1[1] + 48), *(_DWORD *)(a1[1] + 52), a1[2]);
    result = a1[3];
    v11 = (unsigned int)(a2 - 32);
    if ( *(_QWORD *)result )
    {
      v12 = a1[4];
      v13 = a1[2];
      result = *(unsigned int *)(v12 + 8);
      if ( *(_DWORD *)(v13 + 8) <= (unsigned int)result )
      {
        if ( (unsigned int)v11 <= 1 )
          return result;
        goto LABEL_8;
      }
      v28 = *(_QWORD *)v12;
      v29 = (unsigned int)result;
      v30 = *(unsigned int *)(v12 + 12);
      v31 = *(_QWORD *)v12 + 24LL * (unsigned int)result;
      if ( (unsigned int)result >= v30 )
      {
        v132.m128i_i32[0] = a2;
        v70 = &v132;
        v132.m128i_i64[1] = a3;
        v133 = a4;
        if ( v30 < (unsigned __int64)(unsigned int)result + 1 )
        {
          v96 = (const void *)(v12 + 16);
          if ( v28 > (unsigned __int64)&v132 || v31 <= (unsigned __int64)&v132 )
          {
            sub_C8D5F0(v12, v96, (unsigned int)result + 1LL, 0x18u, v11, v28);
            LODWORD(v11) = a2 - 32;
            v29 = *(unsigned int *)(v12 + 8);
            v71 = *(_QWORD *)v12;
          }
          else
          {
            v119 = v28;
            sub_C8D5F0(v12, v96, (unsigned int)result + 1LL, 0x18u, v11, v28);
            LODWORD(v11) = a2 - 32;
            v71 = *(_QWORD *)v12;
            v29 = *(unsigned int *)(v12 + 8);
            v70 = (__m128i *)((char *)&v132 + *(_QWORD *)v12 - v119);
          }
        }
        else
        {
          v71 = *(_QWORD *)v12;
        }
        v72 = (__m128i *)(v71 + 24 * v29);
        *v72 = _mm_loadu_si128(v70);
        v72[1].m128i_i64[0] = v70[1].m128i_i64[0];
        ++*(_DWORD *)(v12 + 8);
      }
      else
      {
        if ( v31 )
        {
          *(_DWORD *)v31 = a2;
          *(_QWORD *)(v31 + 8) = a3;
          *(_QWORD *)(v31 + 16) = a4;
          LODWORD(result) = *(_DWORD *)(v12 + 8);
        }
        *(_DWORD *)(v12 + 8) = result + 1;
      }
      if ( (unsigned int)v11 <= 1 )
        goto LABEL_10;
    }
    else if ( (unsigned int)v11 <= 1 )
    {
      return result;
    }
    v13 = a1[2];
LABEL_8:
    v14 = a1[1];
    if ( BYTE4(a2) )
    {
      v99 = *a1;
      v109 = *(_DWORD *)(v14 + 48);
      v120 = *(_DWORD *)(v14 + 52);
      v15 = sub_B53550(a2);
      sub_27440D0(v99, v15, (unsigned __int8 *)a3, (_BYTE *)a4, v109, v120, v13);
    }
    else
    {
      v27 = *(_QWORD *)(a3 + 8);
      if ( *(_BYTE *)(v27 + 8) == 12 )
      {
        v121 = *a1;
        v100 = *(_DWORD *)(v14 + 48);
        v110 = *(_DWORD *)(v14 + 52);
        switch ( (int)a2 )
        {
          case '"':
          case '#':
            v43 = (_BYTE *)sub_AD64C0(v27, 0, 0);
            v44 = sub_2743410(v121, 0x27u, a3, v43);
            v45 = v121;
            v46 = v110;
            v47 = v100;
            if ( v44 )
              goto LABEL_35;
            v77 = sub_C94E20((__int64)qword_4F862D0);
            v78 = v77 ? *v77 : LODWORD(qword_4F862D0[2]);
            v132 = (__m128i)*(unsigned __int64 *)(v121 + 1264);
            v133 = 0;
            v134 = 0;
            v135 = 0;
            v136 = 0;
            v137 = 0;
            v138 = 0;
            v139 = 257;
            v79 = sub_9AC470(a3, &v132, v78 - 1);
            v45 = v121;
            v46 = v110;
            v47 = v100;
            if ( v79 )
            {
LABEL_35:
              v111 = v47;
              v123 = v46;
              v98 = v45;
              v48 = sub_AD64C0(*(_QWORD *)(a4 + 8), 0, 0);
              v38 = v123;
              v39 = (unsigned __int8 *)a4;
              v97 = v13;
              v40 = v111;
              v41 = (_BYTE *)v48;
              v101 = v123;
              goto LABEL_33;
            }
            break;
          case '$':
          case '%':
            v32 = (_BYTE *)sub_AD64C0(*(_QWORD *)(a4 + 8), 0, 0);
            v33 = sub_2743410(v121, 0x27u, a4, v32);
            v34 = v121;
            v35 = v110;
            v36 = v100;
            if ( v33 )
              goto LABEL_32;
            v73 = sub_C94E20((__int64)qword_4F862D0);
            v74 = v73 ? *v73 : LODWORD(qword_4F862D0[2]);
            v75 = *(_QWORD *)(v121 + 1264);
            v139 = 257;
            v132 = (__m128i)v75;
            v133 = 0;
            v134 = 0;
            v135 = 0;
            v136 = 0;
            v137 = 0;
            v138 = 0;
            v76 = sub_9AC470(a4, &v132, v74 - 1);
            v34 = v121;
            v35 = v110;
            v36 = v100;
            if ( v76 )
            {
LABEL_32:
              v111 = v36;
              v122 = v35;
              v98 = v34;
              v37 = sub_AD64C0(*(_QWORD *)(a4 + 8), 0, 0);
              v38 = v122;
              v39 = (unsigned __int8 *)a3;
              v97 = v13;
              v40 = v111;
              v41 = (_BYTE *)v37;
              v101 = v122;
LABEL_33:
              sub_27440D0(v98, 0x27u, v39, v41, v40, v38, v97);
              v42 = sub_B52E90(a2);
              sub_27440D0(v98, v42, (unsigned __int8 *)a3, (_BYTE *)a4, v111, v101, v13);
            }
            break;
          case '&':
            v59 = (_BYTE *)sub_AD62B0(*(_QWORD *)(a4 + 8));
            v60 = sub_2743410(v121, 0x27u, a4, v59);
            v61 = v121;
            v62 = v110;
            v63 = v100;
            if ( v60 )
            {
              v95 = (_BYTE *)sub_AD64C0(*(_QWORD *)(a4 + 8), 0, 0);
              sub_27440D0(v121, 0x23u, (unsigned __int8 *)a3, v95, v100, v110, v13);
              v63 = v100;
              v62 = v110;
              v61 = v121;
            }
            v102 = v63;
            v112 = v62;
            v124 = v61;
            v64 = (_BYTE *)sub_AD64C0(*(_QWORD *)(a4 + 8), 0, 0);
            v65 = sub_2743410(v124, 0x27u, a4, v64);
            v66 = v124;
            v67 = v112;
            v68 = v102;
            if ( v65 )
              goto LABEL_43;
            v105 = v124;
            v115 = v68;
            v128 = v67;
            v85 = sub_C94E20((__int64)qword_4F862D0);
            v86 = v105;
            v87 = v85 ? *v85 : LODWORD(qword_4F862D0[2]);
            v88 = *(_QWORD *)(v105 + 1264);
            v106 = v115;
            v116 = v128;
            v129 = v86;
            v139 = 257;
            v132 = (__m128i)v88;
            v133 = 0;
            v134 = 0;
            v135 = 0;
            v136 = 0;
            v137 = 0;
            v138 = 0;
            v89 = sub_9AC470(a4, &v132, v87 - 1);
            v66 = v129;
            v67 = v116;
            v68 = v106;
            if ( v89 )
LABEL_43:
              sub_27440D0(v66, 0x22u, (unsigned __int8 *)a3, (_BYTE *)a4, v68, v67, v13);
            break;
          case '\'':
            v54 = (_BYTE *)sub_AD64C0(*(_QWORD *)(a4 + 8), 0, 0);
            v55 = sub_2743410(v121, 0x27u, a4, v54);
            v56 = v121;
            v57 = v110;
            v58 = v100;
            if ( v55 )
              goto LABEL_39;
            v107 = v121;
            v117 = v58;
            v130 = v57;
            v90 = sub_C94E20((__int64)qword_4F862D0);
            v91 = v107;
            v92 = v90 ? *v90 : LODWORD(qword_4F862D0[2]);
            v93 = *(_QWORD *)(v107 + 1264);
            v108 = v117;
            v118 = v130;
            v131 = v91;
            v132 = (__m128i)v93;
            v133 = 0;
            v134 = 0;
            v135 = 0;
            v136 = 0;
            v137 = 0;
            v138 = 0;
            v139 = 257;
            v94 = sub_9AC470(a4, &v132, v92 - 1);
            v56 = v131;
            v57 = v118;
            v58 = v108;
            if ( v94 )
LABEL_39:
              sub_27440D0(v56, 0x23u, (unsigned __int8 *)a3, (_BYTE *)a4, v58, v57, v13);
            break;
          case '(':
            v49 = (_BYTE *)sub_AD64C0(v27, 0, 0);
            v50 = sub_2743410(v121, 0x27u, a3, v49);
            v51 = v121;
            v52 = v110;
            v53 = v100;
            if ( v50 )
              goto LABEL_37;
            v103 = v121;
            v113 = v53;
            v126 = v52;
            v80 = sub_C94E20((__int64)qword_4F862D0);
            v81 = v103;
            v82 = v80 ? *v80 : LODWORD(qword_4F862D0[2]);
            v83 = *(_QWORD *)(v103 + 1264);
            v104 = v113;
            v132 = (__m128i)v83;
            v114 = v126;
            v127 = v81;
            v133 = 0;
            v134 = 0;
            v135 = 0;
            v136 = 0;
            v137 = 0;
            v138 = 0;
            v139 = 257;
            v84 = sub_9AC470(a3, &v132, v82 - 1);
            v51 = v127;
            v52 = v114;
            v53 = v104;
            if ( v84 )
LABEL_37:
              sub_27440D0(v51, 0x24u, (unsigned __int8 *)a3, (_BYTE *)a4, v53, v52, v13);
            break;
          default:
            break;
        }
      }
    }
LABEL_10:
    result = a1[3];
    if ( *(_QWORD *)result )
    {
      v16 = a1[4];
      result = *(unsigned int *)(v16 + 8);
      v17 = *(_DWORD *)(a1[2] + 8);
      if ( (unsigned int)result < v17 )
      {
        v18 = v17 - result;
        v19 = 0;
        v20 = &v132;
        while ( 1 )
        {
          v21 = *(_QWORD *)v16;
          v22 = (unsigned int)result;
          v23 = *(unsigned int *)(v16 + 12);
          v24 = *(_QWORD *)v16 + 24LL * (unsigned int)result;
          if ( (unsigned int)result < v23 )
          {
            if ( v24 )
            {
              *(_DWORD *)v24 = 42;
              *(_QWORD *)(v24 + 8) = 0;
              *(_QWORD *)(v24 + 16) = 0;
              LODWORD(result) = *(_DWORD *)(v16 + 8);
            }
            result = (unsigned int)(result + 1);
            *(_DWORD *)(v16 + 8) = result;
          }
          else
          {
            v25 = (unsigned int)result + 1LL;
            v132 = (__m128i)0x2AuLL;
            v26 = v20;
            v133 = 0;
            if ( v23 < v25 )
            {
              v125 = v20;
              if ( v21 > (unsigned __int64)v20 || v24 <= (unsigned __int64)v20 )
              {
                sub_C8D5F0(v16, (const void *)(v16 + 16), (unsigned int)result + 1LL, 0x18u, (__int64)v20, v25);
                v20 = v125;
                v21 = *(_QWORD *)v16;
                v22 = *(unsigned int *)(v16 + 8);
                v26 = v125;
              }
              else
              {
                v69 = &v20->m128i_i8[-v21];
                sub_C8D5F0(v16, (const void *)(v16 + 16), (unsigned int)result + 1LL, 0x18u, (__int64)v20, v25);
                v21 = *(_QWORD *)v16;
                v22 = *(unsigned int *)(v16 + 8);
                v20 = v125;
                v26 = (const __m128i *)&v69[*(_QWORD *)v16];
              }
            }
            result = v21 + 24 * v22;
            *(__m128i *)result = _mm_loadu_si128(v26);
            *(_QWORD *)(result + 16) = v26[1].m128i_i64[0];
            ++*(_DWORD *)(v16 + 8);
          }
          if ( v18 == ++v19 )
            break;
          v16 = a1[4];
          LODWORD(result) = *(_DWORD *)(v16 + 8);
        }
      }
    }
  }
  return result;
}
