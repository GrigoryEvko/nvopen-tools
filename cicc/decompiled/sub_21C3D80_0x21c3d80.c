// Function: sub_21C3D80
// Address: 0x21c3d80
//
__int64 __fastcall sub_21C3D80(__int64 a1, __int64 a2, double a3, __m128i a4, __m128i a5)
{
  const __m128i *v7; // rax
  __int64 v8; // rsi
  __m128i v9; // xmm0
  __int64 v10; // r14
  __int64 v11; // r13
  _BYTE *v12; // r13
  _QWORD *v13; // rdi
  char v14; // r14
  unsigned int v15; // r13d
  __int64 v16; // r15
  unsigned int v17; // eax
  bool v18; // si
  __int16 v19; // ax
  __int64 v20; // rax
  __m128i v21; // xmm3
  __m128i v22; // xmm4
  __m128i v23; // xmm5
  __m128i v24; // xmm6
  __int64 v25; // r15
  __int64 v26; // rax
  unsigned int v27; // ecx
  unsigned int v28; // edx
  __int64 v29; // r8
  unsigned __int64 v30; // r9
  __int64 v31; // rax
  __m128 *v32; // rax
  __int64 v33; // rdi
  unsigned int v34; // edx
  int v35; // r9d
  __int64 v36; // r13
  unsigned __int64 v37; // r8
  __int64 v38; // rax
  __m128 *v39; // rax
  __int64 v40; // rsi
  __int64 v41; // rdi
  unsigned int v42; // edx
  int v43; // r9d
  __int64 v44; // r13
  __int64 v45; // r8
  __int64 v46; // rax
  __m128 *v47; // rax
  __int64 v48; // rsi
  __int64 v49; // rdi
  unsigned int v50; // edx
  __int64 v51; // r13
  __int64 v52; // r8
  __int64 v53; // rax
  __m128 *v54; // rax
  __int64 v55; // rsi
  __int64 v56; // rdi
  unsigned int v57; // edx
  int v58; // r9d
  __int64 v59; // r13
  __int64 v60; // r8
  __int64 v61; // rax
  __m128 *v62; // rax
  __int64 v63; // rdx
  unsigned int v64; // eax
  unsigned int v65; // r13d
  __int16 v66; // ax
  int v67; // r8d
  __int64 v68; // r9
  __int64 v69; // rax
  __int64 v70; // rax
  _QWORD *v71; // r10
  __int64 v72; // r14
  _QWORD *v73; // rax
  __int64 v74; // rdx
  __int64 v75; // rcx
  __int64 v76; // r8
  __int64 v77; // r9
  __m128 *v78; // rdi
  __int64 v80; // rax
  __int64 v81; // rax
  int v82; // eax
  __int16 v83; // ax
  __int64 v84; // rax
  __int64 v85; // rax
  __int16 v86; // ax
  __int16 v87; // ax
  __int64 v88; // rax
  __int64 v89; // rax
  __int16 v90; // ax
  __int16 v91; // ax
  __int64 v92; // rax
  __int64 v93; // rcx
  __m128 *v94; // rax
  __int128 v95; // [rsp-10h] [rbp-1E0h]
  int v96; // [rsp-8h] [rbp-1D8h]
  __int64 v97; // [rsp+8h] [rbp-1C8h]
  __m128i v98; // [rsp+10h] [rbp-1C0h] BYREF
  unsigned __int64 v99; // [rsp+20h] [rbp-1B0h]
  __int64 v100; // [rsp+28h] [rbp-1A8h]
  __int64 v101; // [rsp+30h] [rbp-1A0h]
  __int64 v102; // [rsp+38h] [rbp-198h]
  __int64 v103; // [rsp+40h] [rbp-190h]
  __int64 v104; // [rsp+48h] [rbp-188h]
  __m128 *v105; // [rsp+50h] [rbp-180h]
  unsigned int v106; // [rsp+5Ch] [rbp-174h]
  __int64 v107; // [rsp+68h] [rbp-168h] BYREF
  __int64 v108; // [rsp+70h] [rbp-160h] BYREF
  __int64 v109; // [rsp+78h] [rbp-158h] BYREF
  __int64 v110; // [rsp+80h] [rbp-150h] BYREF
  unsigned int v111; // [rsp+88h] [rbp-148h] BYREF
  unsigned __int8 v112; // [rsp+8Ch] [rbp-144h]
  __m128i v113; // [rsp+90h] [rbp-140h] BYREF
  __m128i v114; // [rsp+A0h] [rbp-130h] BYREF
  __m128i v115; // [rsp+B0h] [rbp-120h] BYREF
  __int64 v116; // [rsp+C0h] [rbp-110h] BYREF
  int v117; // [rsp+C8h] [rbp-108h]
  __m128 *v118; // [rsp+D0h] [rbp-100h] BYREF
  __int64 v119; // [rsp+D8h] [rbp-F8h]
  __m128 v120; // [rsp+E0h] [rbp-F0h] BYREF
  __m128i v121; // [rsp+F0h] [rbp-E0h]
  __m128i v122; // [rsp+100h] [rbp-D0h]
  __m128i v123; // [rsp+110h] [rbp-C0h]

  v7 = *(const __m128i **)(a2 + 32);
  v8 = *(_QWORD *)(a2 + 72);
  v9 = _mm_loadu_si128(v7);
  v10 = v7[2].m128i_i64[1];
  v113.m128i_i64[0] = 0;
  v113.m128i_i32[2] = 0;
  v11 = v7[3].m128i_u32[0];
  v114.m128i_i64[0] = 0;
  v114.m128i_i32[2] = 0;
  v115.m128i_i64[0] = 0;
  v115.m128i_i32[2] = 0;
  v116 = v8;
  v98 = v9;
  if ( v8 )
    sub_1623A60((__int64)&v116, v8, 2);
  v12 = (_BYTE *)(*(_QWORD *)(v10 + 40) + 16 * v11);
  v13 = *(_QWORD **)(a2 + 104);
  v14 = *(_BYTE *)(a2 + 88);
  v117 = *(_DWORD *)(a2 + 64);
  LOBYTE(v106) = *v12;
  v15 = sub_21BD7A0(v13);
  if ( v15 == 2 )
    sub_16BD130("Cannot store to pointer that points to constant memory space", 1u);
  v16 = sub_1E0A0C0(*(_QWORD *)(*(_QWORD *)(a1 + 272) + 32LL));
  v17 = sub_1E340A0(*(_QWORD *)(a2 + 104));
  LODWORD(v103) = sub_15A9520(v16, v17);
  if ( v15 == 3 || (v18 = 0, v15 <= 1) )
    v18 = (*(_BYTE *)(a2 + 26) & 8) != 0;
  if ( (unsigned __int8)(v14 - 14) <= 0x5Fu )
  {
    switch ( v14 )
    {
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
        v14 = 3;
        break;
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
        v14 = 4;
        break;
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
        v14 = 5;
        break;
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
        v14 = 6;
        break;
      case 55:
        v14 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v14 = 8;
        break;
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
        v14 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v14 = 10;
        break;
      default:
        v14 = 2;
        break;
    }
  }
  v105 = &v120;
  v118 = &v120;
  v119 = 0xC00000000LL;
  v19 = *(_WORD *)(a2 + 24);
  if ( v19 == 665 )
  {
    v80 = *(_QWORD *)(a2 + 32);
    LODWORD(v119) = 2;
    v100 = 2;
    a4 = _mm_loadu_si128((const __m128i *)(v80 + 40));
    a5 = _mm_loadu_si128((const __m128i *)(v80 + 80));
    v25 = *(_QWORD *)(v80 + 120);
    v81 = *(unsigned int *)(v80 + 128);
    v120 = (__m128)a4;
    v104 = v81;
    v121 = a5;
    if ( (_BYTE)v106 != 86 )
    {
LABEL_10:
      v102 = (unsigned int)sub_21BD810(v14);
      v101 = v27;
      goto LABEL_11;
    }
  }
  else
  {
    if ( v19 != 666 )
    {
      v65 = 0;
      goto LABEL_35;
    }
    v20 = *(_QWORD *)(a2 + 32);
    LODWORD(v119) = 4;
    v100 = 4;
    v21 = _mm_loadu_si128((const __m128i *)(v20 + 40));
    v22 = _mm_loadu_si128((const __m128i *)(v20 + 80));
    v23 = _mm_loadu_si128((const __m128i *)(v20 + 120));
    v24 = _mm_loadu_si128((const __m128i *)(v20 + 160));
    v25 = *(_QWORD *)(v20 + 200);
    v26 = *(unsigned int *)(v20 + 208);
    v120 = (__m128)v21;
    v121 = v22;
    v104 = v26;
    v122 = v23;
    v123 = v24;
    if ( (_BYTE)v106 != 86 )
      goto LABEL_10;
  }
  v101 = 3;
  v102 = 32;
  LOBYTE(v106) = 5;
LABEL_11:
  v29 = sub_1D38BB0(*(_QWORD *)(a1 + 272), v18, (__int64)&v116, 5, 0, 1, v9, *(double *)a4.m128i_i64, a5, 0);
  v30 = v28;
  v31 = (unsigned int)v119;
  if ( (unsigned int)v119 >= HIDWORD(v119) )
  {
    v97 = v29;
    v99 = v28;
    sub_16CD150((__int64)&v118, v105, 0, 16, v29, v28);
    v31 = (unsigned int)v119;
    v29 = v97;
    v30 = v99;
  }
  v32 = &v118[v31];
  v32->m128_u64[0] = v29;
  v32->m128_u64[1] = v30;
  v33 = *(_QWORD *)(a1 + 272);
  LODWORD(v119) = v119 + 1;
  v36 = sub_1D38BB0(v33, v15, (__int64)&v116, 5, 0, 1, v9, *(double *)a4.m128i_i64, a5, 0);
  v37 = v34;
  v38 = (unsigned int)v119;
  if ( (unsigned int)v119 >= HIDWORD(v119) )
  {
    v99 = v34;
    sub_16CD150((__int64)&v118, v105, 0, 16, v34, v35);
    v38 = (unsigned int)v119;
    v37 = v99;
  }
  v39 = &v118[v38];
  v40 = v100;
  v39->m128_u64[0] = v36;
  v39->m128_u64[1] = v37;
  v41 = *(_QWORD *)(a1 + 272);
  LODWORD(v119) = v119 + 1;
  v44 = sub_1D38BB0(v41, v40, (__int64)&v116, 5, 0, 1, v9, *(double *)a4.m128i_i64, a5, 0);
  v45 = v42;
  v46 = (unsigned int)v119;
  if ( (unsigned int)v119 >= HIDWORD(v119) )
  {
    v100 = v42;
    sub_16CD150((__int64)&v118, v105, 0, 16, v42, v43);
    v46 = (unsigned int)v119;
    v45 = v100;
  }
  v47 = &v118[v46];
  v48 = v101;
  v47->m128_u64[0] = v44;
  v47->m128_u64[1] = v45;
  v49 = *(_QWORD *)(a1 + 272);
  LODWORD(v119) = v119 + 1;
  v51 = sub_1D38BB0(v49, v48, (__int64)&v116, 5, 0, 1, v9, *(double *)a4.m128i_i64, a5, 0);
  v52 = v50;
  v53 = (unsigned int)v119;
  if ( (unsigned int)v119 >= HIDWORD(v119) )
  {
    v101 = v50;
    sub_16CD150((__int64)&v118, v105, 0, 16, v50, v96);
    v53 = (unsigned int)v119;
    v52 = v101;
  }
  v54 = &v118[v53];
  v55 = v102;
  v54->m128_u64[0] = v51;
  v54->m128_u64[1] = v52;
  v56 = *(_QWORD *)(a1 + 272);
  LODWORD(v119) = v119 + 1;
  v59 = sub_1D38BB0(v56, v55, (__int64)&v116, 5, 0, 1, v9, *(double *)a4.m128i_i64, a5, 0);
  v60 = v57;
  v61 = (unsigned int)v119;
  if ( (unsigned int)v119 >= HIDWORD(v119) )
  {
    v102 = v57;
    sub_16CD150((__int64)&v118, v105, 0, 16, v57, v58);
    v61 = (unsigned int)v119;
    v60 = v102;
  }
  v62 = &v118[v61];
  v62->m128_u64[0] = v59;
  v63 = v104;
  v62->m128_u64[1] = v60;
  LODWORD(v119) = v119 + 1;
  LOBYTE(v64) = sub_21C2A00(a1, v25, v63, (__int64)&v113);
  v65 = v64;
  if ( !(_BYTE)v64 )
  {
    v82 = 8 * v103;
    v102 = (__int64)&v114;
    v103 = (__int64)&v115;
    if ( v82 == 64 )
    {
      if ( (unsigned __int8)sub_21C2BC0(
                              a1,
                              v25,
                              v25,
                              v104,
                              (__int64)&v115,
                              (__int64)&v114,
                              v9,
                              *(double *)a4.m128i_i64,
                              a5) )
      {
LABEL_42:
        v83 = *(_WORD *)(a2 + 24);
        if ( v83 == 665 )
        {
          v110 = 0x100000DC2LL;
          v109 = 0x100000DAALL;
          v108 = 0x100000D9ELL;
          v107 = 0x100000DE6LL;
          sub_21BD570(
            (__int64)&v111,
            v106,
            3570,
            3534,
            3546,
            (__int64)&v107,
            (__int64)&v108,
            (__int64)&v109,
            3510,
            (__int64)&v110);
        }
        else
        {
          if ( v83 != 666 )
          {
LABEL_52:
            v78 = v118;
            goto LABEL_59;
          }
          BYTE4(v110) = 0;
          v109 = 0x100000DB0LL;
          BYTE4(v107) = 0;
          v108 = 0x100000DA4LL;
          sub_21BD570(
            (__int64)&v111,
            v106,
            3576,
            3540,
            3552,
            (__int64)&v107,
            (__int64)&v108,
            (__int64)&v109,
            3516,
            (__int64)&v110);
        }
        v65 = v112;
        v68 = 0;
        v84 = (unsigned int)v119;
        if ( v112 )
          v68 = v111;
        if ( (unsigned int)v119 >= HIDWORD(v119) )
        {
          v106 = v68;
          sub_16CD150((__int64)&v118, v105, 0, 16, v67, v68);
          v84 = (unsigned int)v119;
          v68 = v106;
        }
        v118[v84] = (__m128)_mm_load_si128(&v115);
        v85 = (unsigned int)(v119 + 1);
        LODWORD(v119) = v85;
        if ( HIDWORD(v119) <= (unsigned int)v85 )
        {
          v106 = v68;
          sub_16CD150((__int64)&v118, v105, 0, 16, v67, v68);
          v85 = (unsigned int)v119;
          v68 = v106;
        }
        v118[v85] = (__m128)_mm_load_si128(&v114);
        LODWORD(v119) = v119 + 1;
        if ( (_BYTE)v65 )
          goto LABEL_31;
        goto LABEL_52;
      }
      if ( (unsigned __int8)sub_21C2F80(a1, v25, v25, v104, v103, v102, v9, *(double *)a4.m128i_i64, a5) )
      {
        v87 = *(_WORD *)(a2 + 24);
        if ( v87 == 665 )
        {
          v110 = 0x100000DC1LL;
          v109 = 0x100000DA9LL;
          v108 = 0x100000D9DLL;
          v107 = 0x100000DE5LL;
          sub_21BD570(
            (__int64)&v111,
            v106,
            3569,
            3533,
            3545,
            (__int64)&v107,
            (__int64)&v108,
            (__int64)&v109,
            3509,
            (__int64)&v110);
        }
        else
        {
          if ( v87 != 666 )
            goto LABEL_58;
          BYTE4(v110) = 0;
          v109 = 0x100000DAFLL;
          BYTE4(v107) = 0;
          v108 = 0x100000DA3LL;
          sub_21BD570(
            (__int64)&v111,
            v106,
            3575,
            3539,
            3551,
            (__int64)&v107,
            (__int64)&v108,
            (__int64)&v109,
            3515,
            (__int64)&v110);
        }
LABEL_69:
        v65 = v112;
        v68 = 0;
        v88 = (unsigned int)v119;
        if ( v112 )
          v68 = v111;
        if ( (unsigned int)v119 >= HIDWORD(v119) )
        {
          v106 = v68;
          sub_16CD150((__int64)&v118, v105, 0, 16, v67, v68);
          v88 = (unsigned int)v119;
          v68 = v106;
        }
        v118[v88] = (__m128)_mm_load_si128(&v115);
        v89 = (unsigned int)(v119 + 1);
        LODWORD(v119) = v89;
        if ( HIDWORD(v119) <= (unsigned int)v89 )
        {
          v106 = v68;
          sub_16CD150((__int64)&v118, v105, 0, 16, v67, v68);
          v89 = (unsigned int)v119;
          v68 = v106;
        }
        v118[v89] = (__m128)_mm_load_si128(&v114);
        LODWORD(v119) = v119 + 1;
        goto LABEL_30;
      }
      v91 = *(_WORD *)(a2 + 24);
      if ( v91 == 665 )
      {
        v110 = 0x100000DBFLL;
        v109 = 0x100000DA7LL;
        v108 = 0x100000D9BLL;
        v107 = 0x100000DE3LL;
        sub_21BD570(
          (__int64)&v111,
          v106,
          3567,
          3531,
          3543,
          (__int64)&v107,
          (__int64)&v108,
          (__int64)&v109,
          3507,
          (__int64)&v110);
      }
      else
      {
        if ( v91 != 666 )
          goto LABEL_58;
        BYTE4(v110) = 0;
        v109 = 0x100000DADLL;
        BYTE4(v107) = 0;
        v108 = 0x100000DA1LL;
        sub_21BD570(
          (__int64)&v111,
          v106,
          3573,
          3537,
          3549,
          (__int64)&v107,
          (__int64)&v108,
          (__int64)&v109,
          3513,
          (__int64)&v110);
      }
    }
    else
    {
      if ( (unsigned __int8)sub_21C2BA0(
                              a1,
                              v25,
                              v25,
                              v104,
                              (__int64)&v115,
                              (__int64)&v114,
                              v9,
                              *(double *)a4.m128i_i64,
                              a5) )
        goto LABEL_42;
      if ( (unsigned __int8)sub_21C2F60(a1, v25, v25, v104, v103, v102, v9, *(double *)a4.m128i_i64, a5) )
      {
        v86 = *(_WORD *)(a2 + 24);
        if ( v86 == 665 )
        {
          v110 = 0x100000DC0LL;
          v109 = 0x100000DA8LL;
          v108 = 0x100000D9CLL;
          v107 = 0x100000DE4LL;
          sub_21BD570(
            (__int64)&v111,
            v106,
            3568,
            3532,
            3544,
            (__int64)&v107,
            (__int64)&v108,
            (__int64)&v109,
            3508,
            (__int64)&v110);
        }
        else
        {
          if ( v86 != 666 )
            goto LABEL_58;
          BYTE4(v110) = 0;
          v109 = 0x100000DAELL;
          BYTE4(v107) = 0;
          v108 = 0x100000DA2LL;
          sub_21BD570(
            (__int64)&v111,
            v106,
            3574,
            3538,
            3550,
            (__int64)&v107,
            (__int64)&v108,
            (__int64)&v109,
            3514,
            (__int64)&v110);
        }
        goto LABEL_69;
      }
      v90 = *(_WORD *)(a2 + 24);
      if ( v90 == 665 )
      {
        v110 = 0x100000DBELL;
        v109 = 0x100000DA6LL;
        v108 = 0x100000D9ALL;
        v107 = 0x100000DE2LL;
        sub_21BD570(
          (__int64)&v111,
          v106,
          3566,
          3530,
          3542,
          (__int64)&v107,
          (__int64)&v108,
          (__int64)&v109,
          3506,
          (__int64)&v110);
      }
      else
      {
        if ( v90 != 666 )
          goto LABEL_58;
        BYTE4(v110) = 0;
        v109 = 0x100000DACLL;
        BYTE4(v107) = 0;
        v108 = 0x100000DA0LL;
        sub_21BD570(
          (__int64)&v111,
          v106,
          3572,
          3536,
          3548,
          (__int64)&v107,
          (__int64)&v108,
          (__int64)&v109,
          3512,
          (__int64)&v110);
      }
    }
    v65 = v112;
    v68 = 0;
    v92 = (unsigned int)v119;
    if ( v112 )
      v68 = v111;
    if ( (unsigned int)v119 >= HIDWORD(v119) )
    {
      v106 = v68;
      sub_16CD150((__int64)&v118, v105, 0, 16, v67, v68);
      v92 = (unsigned int)v119;
      v68 = v106;
    }
    v93 = v104;
    v94 = &v118[v92];
    v94->m128_u64[0] = v25;
    v94->m128_u64[1] = v93;
    LODWORD(v119) = v119 + 1;
LABEL_30:
    if ( (_BYTE)v65 )
    {
LABEL_31:
      v70 = (unsigned int)v119;
      if ( (unsigned int)v119 >= HIDWORD(v119) )
      {
        v106 = v68;
        sub_16CD150((__int64)&v118, v105, 0, 16, v67, v68);
        v70 = (unsigned int)v119;
        v68 = v106;
      }
      v118[v70] = (__m128)_mm_load_si128(&v98);
      v71 = *(_QWORD **)(a1 + 272);
      LODWORD(v119) = v119 + 1;
      *((_QWORD *)&v95 + 1) = (unsigned int)v119;
      *(_QWORD *)&v95 = v118;
      v72 = sub_1D2CDB0(v71, v68, (__int64)&v116, 1, 0, v68, v95);
      v73 = (_QWORD *)sub_1E0A240(*(_QWORD *)(a1 + 256), 1);
      *v73 = *(_QWORD *)(a2 + 104);
      *(_QWORD *)(v72 + 88) = v73;
      *(_QWORD *)(v72 + 96) = v73 + 1;
      sub_1D444E0(*(_QWORD *)(a1 + 272), a2, v72);
      sub_1D49010(v72);
      sub_1D2DC70(*(const __m128i **)(a1 + 272), a2, v74, v75, v76, v77);
      v78 = v118;
      goto LABEL_59;
    }
    goto LABEL_52;
  }
  v66 = *(_WORD *)(a2 + 24);
  if ( v66 == 665 )
  {
    v110 = 0x100000DC3LL;
    v109 = 0x100000DABLL;
    v108 = 0x100000D9FLL;
    v107 = 0x100000DE7LL;
    sub_21BD570(
      (__int64)&v111,
      v106,
      3571,
      3535,
      3547,
      (__int64)&v107,
      (__int64)&v108,
      (__int64)&v109,
      3511,
      (__int64)&v110);
    goto LABEL_25;
  }
  if ( v66 == 666 )
  {
    BYTE4(v110) = 0;
    v109 = 0x100000DB1LL;
    BYTE4(v107) = 0;
    v108 = 0x100000DA5LL;
    sub_21BD570(
      (__int64)&v111,
      v106,
      3577,
      3541,
      3553,
      (__int64)&v107,
      (__int64)&v108,
      (__int64)&v109,
      3517,
      (__int64)&v110);
LABEL_25:
    v65 = v112;
    v68 = 0;
    v69 = (unsigned int)v119;
    if ( v112 )
      v68 = v111;
    if ( (unsigned int)v119 >= HIDWORD(v119) )
    {
      v106 = v68;
      sub_16CD150((__int64)&v118, v105, 0, 16, v67, v68);
      v69 = (unsigned int)v119;
      v68 = v106;
    }
    v118[v69] = (__m128)_mm_load_si128(&v113);
    LODWORD(v119) = v119 + 1;
    goto LABEL_30;
  }
LABEL_58:
  v78 = v118;
  v65 = 0;
LABEL_59:
  if ( v78 != v105 )
    _libc_free((unsigned __int64)v78);
LABEL_35:
  if ( v116 )
    sub_161E7C0((__int64)&v116, v116);
  return v65;
}
