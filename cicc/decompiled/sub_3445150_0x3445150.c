// Function: sub_3445150
// Address: 0x3445150
//
unsigned __int8 *__fastcall sub_3445150(
        unsigned __int16 **a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128i a7)
{
  int v7; // r11d
  unsigned __int16 *v13; // r8
  unsigned __int16 v14; // ax
  unsigned __int16 *v15; // rcx
  int v16; // esi
  unsigned __int16 *v17; // rdx
  unsigned int *v18; // rax
  __int64 v19; // rdx
  __int64 v20; // r9
  unsigned __int8 *v22; // r14
  unsigned int v23; // edx
  unsigned __int64 v24; // r15
  int v25; // r9d
  unsigned int v26; // edx
  unsigned __int64 v27; // r13
  __int64 v28; // r9
  unsigned __int8 *v29; // rax
  unsigned __int16 *v30; // r14
  unsigned __int8 *v31; // r12
  unsigned int v32; // edx
  unsigned __int64 v33; // r13
  __int128 v34; // rax
  __int64 v35; // r9
  int v36; // r9d
  __int64 v37; // rdx
  __int64 v38; // rsi
  __int64 v39; // rax
  __int64 v40; // rdx
  unsigned int v41; // esi
  int v42; // eax
  __int64 v43; // rdx
  unsigned __int64 v44; // rax
  unsigned int v45; // r10d
  unsigned __int16 *v46; // rcx
  __int64 v47; // rdx
  __int64 v48; // rax
  char v49; // al
  unsigned __int64 v50; // rax
  unsigned __int64 v51; // rsi
  bool v52; // dl
  unsigned int v53; // eax
  __int64 v54; // r8
  unsigned __int16 *v55; // rdi
  unsigned __int16 v56; // ax
  unsigned int v57; // edx
  unsigned __int64 v58; // r15
  int v59; // r9d
  unsigned int v60; // edx
  unsigned __int64 v61; // r13
  __int64 v62; // r9
  unsigned __int8 *v63; // rax
  unsigned __int16 *v64; // r14
  unsigned __int8 *v65; // r12
  unsigned int v66; // edx
  unsigned __int64 v67; // r13
  __int128 v68; // rax
  __int64 v69; // r9
  __int64 v70; // rdx
  bool v71; // al
  __int64 v72; // rdx
  __int64 v73; // r8
  __int128 v74; // [rsp-20h] [rbp-130h]
  __int128 v75; // [rsp-20h] [rbp-130h]
  __int128 v76; // [rsp-20h] [rbp-130h]
  __int128 v77; // [rsp-20h] [rbp-130h]
  __int128 v78; // [rsp-20h] [rbp-130h]
  __int128 v79; // [rsp-20h] [rbp-130h]
  __int128 v80; // [rsp-10h] [rbp-120h]
  __int128 v81; // [rsp-10h] [rbp-120h]
  __int128 v82; // [rsp-10h] [rbp-120h]
  __int128 v83; // [rsp-10h] [rbp-120h]
  unsigned int v84; // [rsp+0h] [rbp-110h]
  __int64 v85; // [rsp+8h] [rbp-108h]
  __int16 v86; // [rsp+Ah] [rbp-106h]
  unsigned int v87; // [rsp+18h] [rbp-F8h]
  int v88; // [rsp+18h] [rbp-F8h]
  __int64 *v89; // [rsp+20h] [rbp-F0h]
  unsigned int v90; // [rsp+20h] [rbp-F0h]
  __int64 v91; // [rsp+20h] [rbp-F0h]
  __int16 v92; // [rsp+22h] [rbp-EEh]
  __int64 *v93; // [rsp+28h] [rbp-E8h]
  unsigned __int16 *v94; // [rsp+28h] [rbp-E8h]
  __int64 v95; // [rsp+28h] [rbp-E8h]
  unsigned __int8 *v96; // [rsp+50h] [rbp-C0h]
  unsigned __int8 *v97; // [rsp+60h] [rbp-B0h]
  unsigned __int8 *v98; // [rsp+90h] [rbp-80h]
  __int64 v99; // [rsp+B8h] [rbp-58h]
  unsigned __int64 v100; // [rsp+C0h] [rbp-50h] BYREF
  __int64 v101; // [rsp+C8h] [rbp-48h]
  __int64 v102; // [rsp+D0h] [rbp-40h]
  __int64 v103; // [rsp+D8h] [rbp-38h]

  v13 = *a1;
  v14 = **a1;
  if ( !v14 || (v15 = a1[1], v16 = v14, !*(_QWORD *)&v15[4 * v14 + 56]) )
  {
    v22 = sub_33FAF80((__int64)a1[2], 214, (__int64)a1[3], *(unsigned int *)a1[4], *((_QWORD *)a1[4] + 1), a6, a7);
    v24 = v23 | a3 & 0xFFFFFFFF00000000LL;
    v98 = sub_33FAF80((__int64)a1[2], 214, (__int64)a1[3], *(unsigned int *)a1[4], *((_QWORD *)a1[4] + 1), v25, a7);
    v27 = v26 | a5 & 0xFFFFFFFF00000000LL;
    *((_QWORD *)&v81 + 1) = v27;
    *(_QWORD *)&v81 = v98;
    *((_QWORD *)&v75 + 1) = v24;
    *(_QWORD *)&v75 = v22;
    v29 = sub_3406EB0(a1[2], 0x3Au, (__int64)a1[3], *(unsigned int *)a1[4], *((_QWORD *)a1[4] + 1), v28, v75, v81);
    v30 = a1[2];
    v31 = v29;
    v33 = v32 | v27 & 0xFFFFFFFF00000000LL;
    *(_QWORD *)&v34 = sub_3400E40(
                        (__int64)v30,
                        *(unsigned int *)a1[5],
                        *(_DWORD *)a1[4],
                        *((_QWORD *)a1[4] + 1),
                        (__int64)a1[3],
                        a7);
    *((_QWORD *)&v76 + 1) = v33;
    *(_QWORD *)&v76 = v31;
    sub_3406EB0(v30, 0xC0u, (__int64)a1[3], *(unsigned int *)a1[4], *((_QWORD *)a1[4] + 1), v35, v76, v34);
    return sub_33FAF80((__int64)a1[2], 216, (__int64)a1[3], *(unsigned int *)*a1, *((_QWORD *)*a1 + 1), v36, a7);
  }
  v17 = &v15[250 * v14];
  if ( !*(_BYTE *)a1[6] )
  {
    if ( (v17[3293] & 0xFB) != 0 )
    {
      if ( (v17[3239] & 0xFB) == 0 )
        goto LABEL_6;
      goto LABEL_12;
    }
LABEL_10:
    *((_QWORD *)&v82 + 1) = a5;
    *(_QWORD *)&v82 = a4;
    *((_QWORD *)&v77 + 1) = a3;
    *(_QWORD *)&v77 = a2;
    return sub_3406EB0(a1[2], 0xACu, (__int64)a1[3], *(unsigned int *)v13, *((_QWORD *)v13 + 1), a6, v77, v82);
  }
  if ( !*((_BYTE *)v17 + 6586) )
    goto LABEL_10;
  if ( !*((_BYTE *)v17 + 6478) )
  {
LABEL_6:
    v93 = (__int64 *)a1[2];
    v18 = (unsigned int *)sub_33E5110(
                            v93,
                            *(unsigned int *)v13,
                            *((_QWORD *)v13 + 1),
                            *(unsigned int *)v13,
                            *((_QWORD *)v13 + 1));
    *((_QWORD *)&v80 + 1) = a5;
    *(_QWORD *)&v80 = a4;
    *((_QWORD *)&v74 + 1) = a3;
    *(_QWORD *)&v74 = a2;
    return sub_3411F20(v93, 64, (__int64)a1[3], v18, v19, v20, v74, v80);
  }
LABEL_12:
  if ( (unsigned __int16)(v14 - 17) <= 0xD3u )
  {
    v101 = 0;
    v14 = word_4456580[v14 - 1];
    LOWORD(v100) = v14;
    if ( !v14 )
    {
      v92 = HIWORD(v7);
      v94 = v13;
      v39 = sub_3007260((__int64)&v100);
      HIWORD(v7) = v92;
      v13 = v94;
      v102 = v39;
      LODWORD(v38) = v39;
      v103 = v40;
      goto LABEL_19;
    }
    v16 = v14;
  }
  else
  {
    v37 = *((_QWORD *)v13 + 1);
    LOWORD(v100) = **a1;
    v101 = v37;
  }
  if ( v14 == 1 || (unsigned __int16)(v14 - 504) <= 7u )
    BUG();
  v38 = *(_QWORD *)&byte_444C4A0[16 * v16 - 16];
LABEL_19:
  v41 = 2 * v38;
  switch ( v41 )
  {
    case 2u:
      LOWORD(v41) = 3;
      break;
    case 4u:
      LOWORD(v41) = 4;
      break;
    case 8u:
      LOWORD(v41) = 5;
      break;
    case 0x10u:
      LOWORD(v41) = 6;
      break;
    case 0x20u:
      LOWORD(v41) = 7;
      break;
    case 0x40u:
      LOWORD(v41) = 8;
      break;
    case 0x80u:
      LOWORD(v41) = 9;
      break;
    default:
      v42 = sub_3007020(*((_QWORD **)a1[2] + 8), v41);
      v13 = *a1;
      v95 = v43;
      HIWORD(v7) = HIWORD(v42);
      LOWORD(v41) = v42;
      goto LABEL_27;
  }
  v95 = 0;
LABEL_27:
  LODWORD(v44) = *v13;
  LOWORD(v7) = v41;
  HIWORD(v45) = HIWORD(v7);
  if ( (_WORD)v44 )
  {
    if ( (unsigned __int16)(v44 - 17) > 0xD3u )
    {
      if ( *(_BYTE *)a1[7] )
        goto LABEL_56;
      v46 = a1[1];
      v47 = (unsigned __int16)v44;
      if ( !*(_QWORD *)&v46[4 * (unsigned __int16)v44 + 56] )
        goto LABEL_52;
      goto LABEL_31;
    }
    v52 = (unsigned __int16)(v44 - 176) <= 0x34u;
    LODWORD(v51) = word_4456340[(int)v44 - 1];
    LOBYTE(v44) = v52;
  }
  else
  {
    v88 = v7;
    v91 = (__int64)v13;
    v86 = HIWORD(v7);
    if ( !sub_30070B0((__int64)v13) )
    {
      v46 = a1[1];
      HIWORD(v45) = v86;
      goto LABEL_32;
    }
    v50 = sub_3007240(v91);
    v7 = v88;
    v51 = v50;
    v44 = HIDWORD(v50);
    v100 = v51;
    v52 = v44;
  }
  LODWORD(v99) = v51;
  BYTE4(v99) = v44;
  v87 = v7;
  v89 = (__int64 *)*((_QWORD *)a1[2] + 8);
  if ( v52 )
    v53 = sub_2D43AD0(v7, v51);
  else
    v53 = sub_2D43050(v7, v51);
  v41 = v53;
  if ( (_WORD)v53 )
  {
    v95 = 0;
  }
  else
  {
    v84 = sub_3009450(v89, v87, v95, v99, v54, a6);
    v41 = v84;
    v95 = v70;
  }
  v46 = a1[1];
  HIWORD(v45) = HIWORD(v84);
  if ( !*(_BYTE *)a1[7] )
  {
    v55 = *a1;
    LOWORD(v44) = **a1;
    if ( (_WORD)v44 )
    {
      v47 = (unsigned __int16)v44;
      if ( !*(_QWORD *)&v46[4 * (unsigned __int16)v44 + 56] )
      {
LABEL_48:
        if ( (unsigned __int16)(v44 - 17) > 0xD3u )
          goto LABEL_52;
        v56 = word_4456580[(int)v47 - 1];
LABEL_50:
        if ( v56 )
        {
          v47 = v56;
LABEL_52:
          if ( LOBYTE(v46[250 * v47 + 3240]) == 4 )
          {
LABEL_53:
            LOWORD(v45) = v41;
            v90 = v45;
            v97 = sub_33FAF80((__int64)a1[2], 214, (__int64)a1[3], v45, v95, a6, a7);
            v58 = v57 | a3 & 0xFFFFFFFF00000000LL;
            v96 = sub_33FAF80((__int64)a1[2], 214, (__int64)a1[3], v90, v95, v59, a7);
            v61 = v60 | a5 & 0xFFFFFFFF00000000LL;
            *((_QWORD *)&v83 + 1) = v61;
            *(_QWORD *)&v83 = v96;
            *((_QWORD *)&v78 + 1) = v58;
            *(_QWORD *)&v78 = v97;
            v63 = sub_3406EB0(a1[2], 0x3Au, (__int64)a1[3], v90, v95, v62, v78, v83);
            v64 = a1[2];
            v65 = v63;
            v67 = v66 | v61 & 0xFFFFFFFF00000000LL;
            *(_QWORD *)&v68 = sub_3400E40((__int64)v64, *(unsigned int *)a1[5], v90, v95, (__int64)a1[3], a7);
            *((_QWORD *)&v79 + 1) = v67;
            *(_QWORD *)&v79 = v65;
            sub_3406EB0(v64, 0xC0u, (__int64)a1[3], v90, v95, v69, v79, v68);
            return sub_33FAF80((__int64)a1[2], 216, (__int64)a1[3], *(unsigned int *)*a1, *((_QWORD *)*a1 + 1), v36, a7);
          }
        }
LABEL_56:
        v46 = a1[1];
        goto LABEL_32;
      }
LABEL_31:
      if ( LOBYTE(v46[250 * (unsigned int)v47 + 3237]) != 2 )
        goto LABEL_32;
      goto LABEL_48;
    }
    v85 = (__int64)a1[1];
    v71 = sub_30070B0((__int64)v55);
    HIWORD(v45) = HIWORD(v84);
    v46 = (unsigned __int16 *)v85;
    if ( v71 )
    {
      v56 = sub_3009970((__int64)v55, v41, v72, v85, v73);
      v46 = (unsigned __int16 *)v85;
      HIWORD(v45) = HIWORD(v84);
      goto LABEL_50;
    }
  }
LABEL_32:
  v48 = 1;
  if ( (_WORD)v41 == 1 || (_WORD)v41 && (v48 = (unsigned __int16)v41, *(_QWORD *)&v46[4 * (unsigned __int16)v41 + 56]) )
  {
    v49 = v46[250 * v48 + 3236];
    if ( !v49 || v49 == 4 )
      goto LABEL_53;
  }
  return 0;
}
