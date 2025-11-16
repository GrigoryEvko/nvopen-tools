// Function: sub_3283940
// Address: 0x3283940
//
__int64 __fastcall sub_3283940(
        __int64 a1,
        unsigned int a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        int a6,
        __int64 a7,
        __int64 a8,
        int a9,
        __int64 a10)
{
  __int64 v10; // r14
  int v12; // eax
  int v13; // r15d
  int v14; // eax
  __int64 v15; // rdx
  __int64 v16; // r8
  __int64 v17; // rcx
  int v18; // r11d
  int v19; // r9d
  int v20; // esi
  __int64 v21; // rdi
  int v22; // eax
  __int64 v23; // rbx
  __int64 *v24; // rax
  __int64 v25; // rax
  __int64 v26; // rcx
  __int64 v27; // rsi
  __int64 v28; // rdx
  __int64 v29; // rbx
  unsigned int v30; // eax
  unsigned __int64 v31; // rdx
  unsigned __int64 v32; // rdx
  bool v33; // zf
  const void *v34; // rax
  unsigned int v35; // edx
  unsigned __int64 v36; // r15
  bool v37; // al
  unsigned int v38; // r15d
  unsigned __int64 v39; // rax
  int v40; // eax
  __int64 v41; // rax
  unsigned __int16 v42; // dx
  __int64 v43; // rax
  __int16 v44; // ax
  __int64 v45; // rdx
  __int64 v46; // rcx
  unsigned int v47; // r12d
  __int64 v48; // rdi
  bool (__fastcall *v49)(__int64, unsigned int, __int64, __int64, unsigned __int16); // rax
  __int64 v50; // rax
  __int64 v51; // rsi
  unsigned __int16 v52; // bx
  __int64 v53; // rdx
  __int128 v54; // rax
  int v55; // r9d
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // r9
  __int64 v59; // r8
  __int64 result; // rax
  __int64 v61; // rax
  __int64 v62; // rax
  __int64 v63; // rdx
  unsigned __int64 v64; // rsi
  unsigned __int64 v65; // rax
  char v66; // cl
  __int64 v67; // r15
  __int64 *v68; // r12
  unsigned int v69; // ebx
  __int16 v70; // ax
  __int64 v71; // r8
  __int64 v72; // r9
  __int64 v73; // rdx
  __int64 v74; // r8
  __int64 v75; // r9
  __int64 v76; // rdx
  __int64 v77; // rcx
  __int64 v78; // r8
  __int64 v79; // rax
  unsigned __int16 *v80; // r12
  __int64 v81; // rdx
  __int64 v82; // r12
  unsigned int *v83; // rax
  unsigned __int16 *v84; // rax
  __int64 v85; // rdx
  __int64 v86; // rax
  __int64 v87; // rcx
  __int64 *v88; // rsi
  __int64 v89; // rax
  __int64 v90; // rdx
  __int64 v91; // rcx
  __int64 v92; // r8
  unsigned int *v93; // rax
  __int64 v94; // rdx
  __int64 v95; // r15
  char v96; // dl
  unsigned __int64 v97; // rax
  int v98; // eax
  unsigned __int64 v99; // rax
  int v100; // eax
  __int64 v101; // rdx
  __int64 v102; // rcx
  __int64 v103; // r8
  __int16 v104; // ax
  bool v105; // al
  __int64 v106; // rcx
  __int16 v107; // ax
  unsigned int v108; // [rsp+Ch] [rbp-9Ch]
  int v109; // [rsp+Ch] [rbp-9Ch]
  bool v110; // [rsp+Ch] [rbp-9Ch]
  unsigned int v111; // [rsp+Ch] [rbp-9Ch]
  unsigned int v112; // [rsp+Ch] [rbp-9Ch]
  const void **v113; // [rsp+10h] [rbp-98h]
  unsigned int v114; // [rsp+10h] [rbp-98h]
  unsigned int v115; // [rsp+18h] [rbp-90h]
  char v116; // [rsp+20h] [rbp-88h]
  __int64 v117; // [rsp+20h] [rbp-88h]
  __int64 v118; // [rsp+28h] [rbp-80h] BYREF
  __int64 v119; // [rsp+30h] [rbp-78h]
  unsigned __int64 v120; // [rsp+38h] [rbp-70h] BYREF
  unsigned int v121; // [rsp+40h] [rbp-68h]
  const void *v122; // [rsp+48h] [rbp-60h] BYREF
  __int64 v123; // [rsp+50h] [rbp-58h]
  __int64 v124; // [rsp+58h] [rbp-50h] BYREF
  __int64 v125; // [rsp+60h] [rbp-48h]
  __int64 v126; // [rsp+68h] [rbp-40h] BYREF
  __int64 v127; // [rsp+70h] [rbp-38h]

  v10 = a1;
  v119 = a4;
  v118 = a3;
  v12 = sub_32715E0(a1, a2, a3, a4, a5, a6, a7, a8, a9);
  if ( !v12 )
    return 0;
  v13 = v12;
  v14 = *(_DWORD *)(a1 + 24);
  if ( v13 == 181 && v14 == 226 )
  {
    v116 = sub_33E0720(a7, a8, 0);
    if ( !v116 )
    {
LABEL_84:
      v14 = *(_DWORD *)(a1 + 24);
      goto LABEL_4;
    }
    v80 = (unsigned __int16 *)(*(_QWORD *)(a1 + 48) + 16LL * a2);
    LODWORD(v81) = *v80;
    v82 = *((_QWORD *)v80 + 1);
    LOWORD(v126) = v81;
    v127 = v82;
    if ( (_WORD)v81 )
    {
      if ( (unsigned __int16)(v81 - 17) <= 0xD3u )
      {
        v82 = 0;
        LOWORD(v81) = word_4456580[(unsigned __int16)v81 - 1];
      }
    }
    else
    {
      v112 = v81;
      v105 = sub_30070B0((__int64)&v126);
      LOWORD(v81) = v112;
      if ( v105 )
      {
        v107 = sub_3009970((__int64)&v126, a8, v112, v106, v74);
        v82 = v81;
        LOWORD(v81) = v107;
      }
    }
    v83 = *(unsigned int **)(a1 + 40);
    LOWORD(v124) = v81;
    v125 = v82;
    v84 = (unsigned __int16 *)(*(_QWORD *)(*(_QWORD *)v83 + 48LL) + 16LL * v83[2]);
    v85 = *v84;
    v86 = *((_QWORD *)v84 + 1);
    LOWORD(v122) = v85;
    v123 = v86;
    if ( (_WORD)v85 )
    {
      v87 = (unsigned int)(v85 - 17);
      if ( (unsigned __int16)(v85 - 17) > 0xD3u )
      {
        LOWORD(v126) = v85;
        v127 = v86;
        goto LABEL_111;
      }
      v104 = word_4456580[(unsigned __int16)v85 - 1];
      v85 = 0;
    }
    else
    {
      if ( !sub_30070B0((__int64)&v122) )
        goto LABEL_84;
      v104 = sub_3009970((__int64)&v122, a8, v101, v102, v103);
    }
    LOWORD(v126) = v104;
    v127 = v85;
    if ( !v104 )
      goto LABEL_84;
LABEL_111:
    v88 = *(__int64 **)(a10 + 64);
    v89 = sub_3007410((__int64)&v126, v88, v85, v87, v74, v75);
    v93 = (unsigned int *)sub_BCAC60(v89, (__int64)v88, v90, v91, v92);
    v114 = sub_C336E0(v93, 1);
    v122 = (const void *)sub_2D5B750((unsigned __int16 *)&v124);
    v123 = v94;
    if ( sub_CA1930(&v122) >= (unsigned __int64)v114 )
    {
      LODWORD(v95) = 0;
      if ( v114 )
      {
        v96 = 64;
        if ( v114 != 1 )
        {
          _BitScanReverse64(&v97, v114 - 1LL);
          v96 = v97 ^ 0x3F;
        }
        v95 = 1LL << (64 - v96);
      }
      v115 = v95;
      goto LABEL_39;
    }
    goto LABEL_84;
  }
LABEL_4:
  v124 = 0;
  LODWORD(v125) = 0;
  if ( v14 > 206 )
  {
    if ( v14 != 207 )
      return 0;
    v62 = *(_QWORD *)(a1 + 40);
    v21 = *(_QWORD *)v62;
    v20 = *(_DWORD *)(v62 + 8);
    v124 = *(_QWORD *)(v62 + 40);
    LODWORD(v125) = *(_DWORD *)(v62 + 48);
    v10 = *(_QWORD *)(v62 + 80);
    v19 = *(_DWORD *)(v62 + 88);
    v16 = *(_QWORD *)(v62 + 120);
    v17 = *(unsigned int *)(v62 + 128);
    v18 = *(_DWORD *)(*(_QWORD *)(v62 + 160) + 96LL);
  }
  else if ( v14 > 204 )
  {
    v61 = *(_QWORD *)(a1 + 40);
    if ( *(_DWORD *)(*(_QWORD *)v61 + 24LL) != 208 )
      return 0;
    v63 = *(_QWORD *)(*(_QWORD *)v61 + 40LL);
    v21 = *(_QWORD *)v63;
    v20 = *(_DWORD *)(v63 + 8);
    v124 = *(_QWORD *)(v63 + 40);
    LODWORD(v125) = *(_DWORD *)(v63 + 48);
    v10 = *(_QWORD *)(v61 + 40);
    v19 = *(_DWORD *)(v61 + 48);
    v16 = *(_QWORD *)(v61 + 80);
    v17 = *(unsigned int *)(v61 + 88);
    v18 = *(_DWORD *)(*(_QWORD *)(v63 + 80) + 96LL);
  }
  else
  {
    if ( (unsigned int)(v14 - 180) > 1 )
      return 0;
    v15 = *(_QWORD *)(a1 + 40);
    v16 = *(_QWORD *)(v15 + 40);
    v17 = *(unsigned int *)(v15 + 48);
    v18 = 2 * (v14 == 180) + 18;
    v10 = *(_QWORD *)v15;
    v19 = *(_DWORD *)(v15 + 8);
    v124 = v16;
    LODWORD(v125) = v17;
    v20 = v19;
    v21 = v10;
  }
  v22 = sub_32715E0(v21, v20, v124, v125, v10, v19, v16, v17, v18);
  if ( !v22 || v13 == v22 )
    return 0;
  if ( v13 == 180 )
  {
    v23 = sub_33DFBC0(v118, v119, 0, 0);
    v24 = &v124;
  }
  else
  {
    v23 = sub_33DFBC0(v124, v125, 0, 0);
    v24 = &v118;
  }
  v25 = sub_33DFBC0(*v24, v24[1], 0, 0);
  if ( !v23 )
    return 0;
  if ( !v25 )
    return 0;
  v26 = *(_QWORD *)(v23 + 48);
  v27 = *(_QWORD *)(v25 + 48);
  if ( *(_WORD *)v27 != *(_WORD *)v26 )
    return 0;
  v116 = *(_WORD *)v26 == 0 && *(_QWORD *)(v27 + 8) != *(_QWORD *)(v26 + 8);
  if ( v116 )
    return 0;
  v28 = *(_QWORD *)(v23 + 96);
  v29 = *(_QWORD *)(v25 + 96);
  v113 = (const void **)(v29 + 24);
  LODWORD(v123) = *(_DWORD *)(v28 + 32);
  if ( (unsigned int)v123 > 0x40 )
    sub_C43780((__int64)&v122, (const void **)(v28 + 24));
  else
    v122 = *(const void **)(v28 + 24);
  sub_C46A40((__int64)&v122, 1);
  LODWORD(v127) = v123;
  v126 = (__int64)v122;
  v30 = *(_DWORD *)(v29 + 32);
  LODWORD(v123) = v30;
  if ( v30 > 0x40 )
  {
    sub_C43780((__int64)&v122, v113);
    v30 = v123;
    if ( (unsigned int)v123 > 0x40 )
    {
      sub_C43D10((__int64)&v122);
      goto LABEL_23;
    }
    v31 = (unsigned __int64)v122;
  }
  else
  {
    v31 = *(_QWORD *)(v29 + 24);
  }
  v32 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v30) & ~v31;
  v33 = v30 == 0;
  v34 = 0;
  if ( !v33 )
    v34 = (const void *)v32;
  v122 = v34;
LABEL_23:
  sub_C46250((__int64)&v122);
  v35 = v123;
  v36 = (unsigned __int64)v122;
  LODWORD(v123) = 0;
  v121 = v35;
  v120 = (unsigned __int64)v122;
  if ( v35 <= 0x40 )
  {
    if ( (const void *)v126 != v122 )
    {
      v38 = v127;
LABEL_96:
      if ( *(_DWORD *)(v29 + 32) > 0x40u )
      {
        v109 = *(_DWORD *)(v29 + 32);
        if ( v109 - (unsigned int)sub_C444A0((__int64)v113) > 0x40 )
          goto LABEL_99;
        v79 = **(_QWORD **)(v29 + 24);
      }
      else
      {
        v79 = *(_QWORD *)(v29 + 24);
      }
      if ( !v79 )
      {
        if ( v38 > 0x40 )
        {
          if ( (unsigned int)sub_C44630((__int64)&v126) != 1 )
          {
            v10 = 0;
            goto LABEL_36;
          }
          v100 = sub_C444A0((__int64)&v126);
        }
        else
        {
          if ( !v126 || (v126 & (v126 - 1)) != 0 )
            return 0;
          _BitScanReverse64(&v99, v126);
          v100 = v38 + (v99 ^ 0x3F) - 64;
        }
        v116 = 1;
        v115 = v38 - 1 - v100;
LABEL_35:
        if ( v38 <= 0x40 )
          goto LABEL_38;
        goto LABEL_36;
      }
LABEL_99:
      v10 = 0;
      goto LABEL_35;
    }
  }
  else
  {
    v108 = v35;
    v37 = sub_C43C50((__int64)&v120, (const void **)&v126);
    v35 = v108;
    if ( !v37 )
    {
LABEL_100:
      if ( v36 )
      {
        v110 = v37;
        j_j___libc_free_0_0(v36);
        v37 = v110;
        if ( (unsigned int)v123 > 0x40 )
        {
          if ( v122 )
          {
            j_j___libc_free_0_0((unsigned __int64)v122);
            v37 = v110;
          }
        }
      }
      goto LABEL_29;
    }
  }
  if ( (unsigned int)v127 > 0x40 )
  {
    v111 = v35;
    v98 = sub_C44630((__int64)&v126);
    v35 = v111;
    v37 = v98 == 1;
  }
  else
  {
    v37 = 0;
    if ( v126 )
      v37 = (v126 & (v126 - 1)) == 0;
  }
  if ( v35 > 0x40 )
    goto LABEL_100;
LABEL_29:
  v38 = v127;
  if ( !v37 )
    goto LABEL_96;
  if ( (unsigned int)v127 <= 0x40 )
  {
    v115 = 0;
    if ( !v126 || (v126 & (v126 - 1)) != 0 )
      goto LABEL_38;
    _BitScanReverse64(&v39, v126);
    v40 = v127 + (v39 ^ 0x3F) - 64;
    goto LABEL_34;
  }
  if ( (unsigned int)sub_C44630((__int64)&v126) == 1 )
  {
    v40 = sub_C444A0((__int64)&v126);
LABEL_34:
    v115 = v38 - v40;
    goto LABEL_35;
  }
  v115 = 0;
LABEL_36:
  if ( v126 )
    j_j___libc_free_0_0(v126);
LABEL_38:
  if ( !v10 )
    return 0;
LABEL_39:
  if ( *(_DWORD *)(v10 + 24) != 226 )
    return 0;
  v41 = *(_QWORD *)(**(_QWORD **)(v10 + 40) + 48LL) + 16LL * *(unsigned int *)(*(_QWORD *)(v10 + 40) + 8LL);
  v42 = *(_WORD *)v41;
  v43 = *(_QWORD *)(v41 + 8);
  LOWORD(v122) = v42;
  v123 = v43;
  switch ( v115 )
  {
    case 1u:
      v44 = 2;
      v46 = 0;
      break;
    case 2u:
      v44 = 3;
      v46 = 0;
      break;
    case 4u:
      v44 = 4;
      v46 = 0;
      break;
    case 8u:
      v44 = 5;
      v46 = 0;
      break;
    case 0x10u:
      v44 = 6;
      v46 = 0;
      break;
    case 0x20u:
      v44 = 7;
      v46 = 0;
      break;
    case 0x40u:
      v44 = 8;
      v46 = 0;
      break;
    case 0x80u:
      v44 = 9;
      v46 = 0;
      break;
    default:
      v44 = sub_3007020(*(_QWORD **)(a10 + 64), v115);
      v46 = v45;
      v42 = (unsigned __int16)v122;
      break;
  }
  LOWORD(v124) = v44;
  v125 = v46;
  if ( v42 )
  {
    if ( (unsigned __int16)(v42 - 17) > 0xD3u )
      goto LABEL_51;
    v66 = (unsigned __int16)(v42 - 176) <= 0x34u;
    LODWORD(v64) = word_4456340[v42 - 1];
    LOBYTE(v65) = v66;
  }
  else
  {
    if ( !sub_30070B0((__int64)&v122) )
      goto LABEL_51;
    v64 = sub_3007240((__int64)&v122);
    v65 = HIDWORD(v64);
    v120 = v64;
    v66 = BYTE4(v64);
  }
  v67 = v125;
  LODWORD(v126) = v64;
  BYTE4(v126) = v65;
  v68 = *(__int64 **)(a10 + 64);
  v69 = v124;
  if ( v66 )
    v70 = sub_2D43AD0(v124, v64);
  else
    v70 = sub_2D43050(v124, v64);
  v73 = 0;
  if ( !v70 )
    v70 = sub_3009450(v68, v69, v67, v126, v71, v72);
  LOWORD(v124) = v70;
  v125 = v73;
LABEL_51:
  v47 = 228 - ((v116 == 0) - 1);
  v48 = *(_QWORD *)(a10 + 16);
  v49 = *(bool (__fastcall **)(__int64, unsigned int, __int64, __int64, unsigned __int16))(*(_QWORD *)v48 + 1752LL);
  if ( v49 != sub_2FE3620 )
  {
    if ( ((unsigned __int8 (__fastcall *)(__int64, _QWORD, _QWORD, __int64, _QWORD, __int64))v49)(
           v48,
           v47,
           (unsigned int)v122,
           v123,
           (unsigned int)v124,
           v125) )
    {
      goto LABEL_54;
    }
    return 0;
  }
  v50 = 1;
  if ( (_WORD)v124 != 1 )
  {
    if ( !(_WORD)v124 )
      return 0;
    v50 = (unsigned __int16)v124;
    if ( !*(_QWORD *)(v48 + 8LL * (unsigned __int16)v124 + 112) )
      return 0;
  }
  if ( (*(_BYTE *)(v47 + v48 + 500 * v50 + 6414) & 0xFB) != 0 )
    return 0;
LABEL_54:
  v51 = *(_QWORD *)(v10 + 80);
  v126 = v51;
  if ( v51 )
    sub_B96E90((__int64)&v126, v51, 1);
  v52 = v124;
  LODWORD(v127) = *(_DWORD *)(v10 + 72);
  if ( (_WORD)v124 )
  {
    if ( (unsigned __int16)(v124 - 17) <= 0xD3u )
    {
      v52 = word_4456580[(unsigned __int16)v124 - 1];
      v53 = 0;
      goto LABEL_59;
    }
  }
  else if ( sub_30070B0((__int64)&v124) )
  {
    v52 = sub_3009970((__int64)&v124, v51, v76, v77, v78);
    goto LABEL_59;
  }
  v53 = v125;
LABEL_59:
  *(_QWORD *)&v54 = sub_33F7D60(a10, v52, v53);
  v56 = sub_3406EB0(a10, v47, (unsigned int)&v126, v124, v125, v55, *(_OWORD *)*(_QWORD *)(v10 + 40), v54);
  v58 = *(_QWORD *)(*(_QWORD *)(a5 + 48) + 8LL);
  v59 = **(unsigned __int16 **)(a5 + 48);
  if ( v116 )
    result = sub_33FB310(a10, v56, v57, &v126, v59, v58);
  else
    result = sub_33FB160(a10, v56, v57, &v126, v59, v58);
  if ( v126 )
  {
    v117 = result;
    sub_B91220((__int64)&v126, v126);
    return v117;
  }
  return result;
}
