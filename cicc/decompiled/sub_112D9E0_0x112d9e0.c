// Function: sub_112D9E0
// Address: 0x112d9e0
//
unsigned __int8 *__fastcall sub_112D9E0(const __m128i *a1, __int64 a2, __int64 a3, _BYTE *a4, __int64 a5)
{
  char v5; // r15
  unsigned int v6; // r14d
  __int64 v10; // rdx
  __int64 v11; // rax
  unsigned int v12; // eax
  int v13; // edi
  __int64 v14; // rax
  unsigned int v15; // eax
  int v16; // edi
  __m128i v17; // xmm4
  __m128i v18; // xmm5
  unsigned __int64 v19; // xmm6_8
  __m128i v20; // xmm7
  __m128i v21; // xmm4
  __m128i v22; // xmm5
  unsigned __int64 v23; // xmm6_8
  __m128i v24; // xmm7
  __m128i v25; // xmm2
  __m128i v26; // xmm0
  __m128i v27; // xmm1
  __m128i v28; // xmm3
  unsigned __int64 v29; // rbx
  unsigned __int8 *v30; // rax
  __int64 v31; // r9
  unsigned __int8 v32; // al
  unsigned int v33; // r15d
  __int64 v34; // rdi
  int v35; // eax
  bool v36; // al
  __int64 v37; // r8
  char v38; // r9
  char v39; // r10
  unsigned __int8 v40; // al
  unsigned int v41; // ebx
  int v42; // eax
  _BYTE *v43; // rax
  char v44; // bl
  __int64 v45; // rax
  unsigned int v46; // eax
  unsigned __int64 v47; // rdi
  __int64 v48; // rax
  unsigned __int16 v49; // ax
  __int64 v50; // r9
  char v51; // si
  __int64 v53; // rax
  unsigned int v54; // eax
  int v55; // r13d
  int v56; // eax
  __int64 v57; // rax
  bool v58; // al
  char v59; // al
  __int64 v60; // r15
  __int64 v61; // rdx
  _BYTE *v62; // rax
  bool v63; // r15
  unsigned int i; // ecx
  __int64 v65; // rax
  unsigned int v66; // ecx
  unsigned int v67; // r15d
  int v68; // eax
  unsigned __int8 *v69; // rax
  __int64 v70; // rbx
  __int64 v71; // rdx
  _BYTE *v72; // rax
  unsigned int v73; // ebx
  int v74; // eax
  __int64 v75; // rax
  unsigned int v76; // eax
  unsigned int v77; // edi
  int v78; // eax
  __int64 v79; // r9
  __int16 v80; // si
  bool v81; // bl
  unsigned int j; // ecx
  __int64 v83; // rax
  unsigned int v84; // ecx
  unsigned int v85; // ebx
  int v86; // eax
  __int64 v87; // rax
  __int64 v88; // rax
  int v89; // [rsp+8h] [rbp-118h]
  int v90; // [rsp+Ch] [rbp-114h]
  unsigned int v91; // [rsp+Ch] [rbp-114h]
  char v92; // [rsp+Ch] [rbp-114h]
  __int64 v93; // [rsp+10h] [rbp-110h]
  unsigned __int8 v94; // [rsp+10h] [rbp-110h]
  unsigned int v95; // [rsp+10h] [rbp-110h]
  char v96; // [rsp+10h] [rbp-110h]
  char v97; // [rsp+10h] [rbp-110h]
  char v98; // [rsp+10h] [rbp-110h]
  unsigned int v99; // [rsp+10h] [rbp-110h]
  __int64 v100; // [rsp+18h] [rbp-108h]
  char v101; // [rsp+18h] [rbp-108h]
  char v102; // [rsp+18h] [rbp-108h]
  __int64 v103; // [rsp+18h] [rbp-108h]
  char v104; // [rsp+18h] [rbp-108h]
  char v105; // [rsp+18h] [rbp-108h]
  char v107; // [rsp+28h] [rbp-F8h]
  char v108; // [rsp+28h] [rbp-F8h]
  char v109; // [rsp+28h] [rbp-F8h]
  __int64 v110; // [rsp+28h] [rbp-F8h]
  __int64 v111; // [rsp+28h] [rbp-F8h]
  _BYTE *v112; // [rsp+30h] [rbp-F0h]
  _BYTE *v113; // [rsp+38h] [rbp-E8h]
  __m128i v115[2]; // [rsp+50h] [rbp-D0h] BYREF
  unsigned __int64 v116; // [rsp+70h] [rbp-B0h]
  __int64 v117; // [rsp+78h] [rbp-A8h]
  __m128i v118; // [rsp+80h] [rbp-A0h]
  __int64 v119; // [rsp+90h] [rbp-90h]
  __m128i v120; // [rsp+A0h] [rbp-80h] BYREF
  __m128i v121; // [rsp+B0h] [rbp-70h]
  __m128i v122; // [rsp+C0h] [rbp-60h]
  __m128i v123; // [rsp+D0h] [rbp-50h]
  __int64 v124; // [rsp+E0h] [rbp-40h]

  v6 = a5;
  v10 = *(_DWORD *)(a3 + 4) & 0x7FFFFFF;
  v112 = *(_BYTE **)(a3 - 32 * v10);
  v113 = *(_BYTE **)(a3 + 32 * (1 - v10));
  if ( sub_B532B0(a5) )
  {
    v11 = *(_QWORD *)(a3 - 32);
    if ( !v11 || *(_BYTE *)v11 || *(_QWORD *)(v11 + 24) != *(_QWORD *)(a3 + 80) )
      BUG();
    v12 = *(_DWORD *)(v11 + 36);
    if ( v12 == 365 )
    {
      v13 = 34;
    }
    else if ( v12 > 0x16D )
    {
      if ( v12 != 366 )
        goto LABEL_172;
      v13 = 36;
    }
    else if ( v12 == 329 )
    {
      v13 = 38;
    }
    else
    {
      if ( v12 != 330 )
        goto LABEL_172;
      v13 = 40;
    }
    if ( !sub_B532B0(v13) )
      return 0;
  }
  if ( sub_B532A0(v6) )
  {
    v14 = *(_QWORD *)(a3 - 32);
    if ( !v14 || *(_BYTE *)v14 || *(_QWORD *)(v14 + 24) != *(_QWORD *)(a3 + 80) )
      BUG();
    v15 = *(_DWORD *)(v14 + 36);
    if ( v15 == 365 )
    {
      v16 = 34;
    }
    else if ( v15 > 0x16D )
    {
      if ( v15 != 366 )
        goto LABEL_172;
      v16 = 36;
    }
    else if ( v15 == 329 )
    {
      v16 = 38;
    }
    else
    {
      if ( v15 != 330 )
        goto LABEL_172;
      v16 = 40;
    }
    if ( sub_B532B0(v16) )
    {
      v17 = _mm_loadu_si128(a1 + 6);
      v18 = _mm_loadu_si128(a1 + 7);
      v119 = a1[10].m128i_i64[0];
      v19 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
      v115[0] = v17;
      v20 = _mm_loadu_si128(a1 + 9);
      v115[1] = v18;
      v116 = v19;
      v118 = v20;
      v117 = a2;
      if ( !(unsigned __int8)sub_9AC470((__int64)a4, v115, 0) )
        return 0;
      v21 = _mm_loadu_si128(a1 + 6);
      v22 = _mm_loadu_si128(a1 + 7);
      v23 = _mm_loadu_si128(a1 + 8).m128i_u64[0];
      v124 = a1[10].m128i_i64[0];
      v24 = _mm_loadu_si128(a1 + 9);
      v120 = v21;
      v122.m128i_i64[0] = v23;
      v121 = v22;
      v122.m128i_i64[1] = a2;
      v123 = v24;
      if ( !(unsigned __int8)sub_9AC470(a3, &v120, 0) )
        return 0;
      v6 = sub_B53550(v6);
    }
  }
  v25 = _mm_loadu_si128(a1 + 8);
  v26 = _mm_loadu_si128(a1 + 6);
  v27 = _mm_loadu_si128(a1 + 7);
  v124 = a1[10].m128i_i64[0];
  v122 = v25;
  v28 = _mm_loadu_si128(a1 + 9);
  v122.m128i_i64[1] = a2;
  v120 = v26;
  v29 = v6 | a5 & 0xFFFFFF0000000000LL;
  v121 = v27;
  v123 = v28;
  v30 = (unsigned __int8 *)sub_1016CC0(v29, v112, a4, v120.m128i_i64);
  v31 = (__int64)v30;
  if ( !v30 )
  {
    v37 = sub_1016CC0(v29, v113, a4, v120.m128i_i64);
    if ( !v37 )
      return 0;
    v38 = 0;
    v39 = 0;
    goto LABEL_31;
  }
  v32 = *v30;
  if ( v32 == 17 )
  {
    v33 = *(_DWORD *)(v31 + 32);
    if ( v33 > 0x40 )
    {
      v100 = v31;
      v34 = v31 + 24;
LABEL_27:
      v35 = sub_C444A0(v34);
      v31 = v100;
      v36 = v33 - 1 == v35;
      goto LABEL_28;
    }
    v36 = *(_QWORD *)(v31 + 24) == 1;
    goto LABEL_28;
  }
  v60 = *(_QWORD *)(v31 + 8);
  v61 = (unsigned int)*(unsigned __int8 *)(v60 + 8) - 17;
  if ( (unsigned int)v61 > 1 || v32 > 0x15u )
  {
LABEL_92:
    v96 = sub_1112510(v31);
    v69 = (unsigned __int8 *)sub_1016CC0(v29, v113, a4, v120.m128i_i64);
    v38 = v96;
    v37 = (__int64)v69;
    if ( !v69 )
    {
      if ( !v96 )
        return 0;
      v44 = 0;
      v38 = 1;
      v5 = 0;
      goto LABEL_37;
    }
    v40 = *v69;
    v39 = v96;
    v5 = 0;
    if ( v40 == 17 )
      goto LABEL_32;
    goto LABEL_94;
  }
  v100 = v31;
  v62 = sub_AD7630(v31, 0, v61);
  v31 = v100;
  if ( !v62 || *v62 != 17 )
  {
    if ( *(_BYTE *)(v60 + 8) == 17 )
    {
      v90 = *(_DWORD *)(v60 + 32);
      if ( v90 )
      {
        v63 = 0;
        for ( i = 0; i != v90; i = v66 + 1 )
        {
          v103 = v31;
          v95 = i;
          v65 = sub_AD69F0((unsigned __int8 *)v31, i);
          v31 = v103;
          if ( !v65 )
            goto LABEL_92;
          v66 = v95;
          if ( *(_BYTE *)v65 != 13 )
          {
            if ( *(_BYTE *)v65 != 17 )
              goto LABEL_92;
            v67 = *(_DWORD *)(v65 + 32);
            if ( v67 <= 0x40 )
            {
              v63 = *(_QWORD *)(v65 + 24) == 1;
            }
            else
            {
              v68 = sub_C444A0(v65 + 24);
              v31 = v103;
              v66 = v95;
              v63 = v67 - 1 == v68;
            }
            if ( !v63 )
              goto LABEL_92;
          }
        }
        if ( v63 )
          goto LABEL_29;
      }
    }
    goto LABEL_92;
  }
  v33 = *((_DWORD *)v62 + 8);
  if ( v33 > 0x40 )
  {
    v34 = (__int64)(v62 + 24);
    goto LABEL_27;
  }
  v36 = *((_QWORD *)v62 + 3) == 1;
LABEL_28:
  if ( !v36 )
    goto LABEL_92;
LABEL_29:
  v37 = sub_1016CC0(v29, v113, a4, v120.m128i_i64);
  if ( !v37 )
  {
    v44 = 0;
    v38 = 1;
    v5 = 1;
    goto LABEL_37;
  }
  v5 = 1;
  v38 = 1;
  v39 = 1;
LABEL_31:
  v40 = *(_BYTE *)v37;
  if ( *(_BYTE *)v37 == 17 )
  {
LABEL_32:
    v41 = *(_DWORD *)(v37 + 32);
    if ( v41 > 0x40 )
    {
      v101 = v39;
      v108 = v38;
      v93 = v37;
      v42 = sub_C444A0(v37 + 24);
      v37 = v93;
      v38 = v108;
      v39 = v101;
      if ( v42 == v41 - 1 )
        goto LABEL_34;
      goto LABEL_70;
    }
    v58 = *(_QWORD *)(v37 + 24) == 1;
    goto LABEL_69;
  }
LABEL_94:
  v70 = *(_QWORD *)(v37 + 8);
  v71 = (unsigned int)*(unsigned __int8 *)(v70 + 8) - 17;
  if ( (unsigned int)v71 > 1 || v40 > 0x15u )
    goto LABEL_70;
  v97 = v39;
  v104 = v38;
  v110 = v37;
  v72 = sub_AD7630(v37, 0, v71);
  v37 = v110;
  v38 = v104;
  v39 = v97;
  if ( !v72 || *v72 != 17 )
  {
    if ( *(_BYTE *)(v70 + 8) != 17 )
      goto LABEL_70;
    v89 = *(_DWORD *)(v70 + 32);
    if ( !v89 )
      goto LABEL_70;
    v81 = 0;
    for ( j = 0; j != v89; j = v84 + 1 )
    {
      v98 = v39;
      v105 = v38;
      v111 = v37;
      v91 = j;
      v83 = sub_AD69F0((unsigned __int8 *)v37, j);
      v37 = v111;
      v38 = v105;
      v39 = v98;
      if ( !v83 )
        goto LABEL_70;
      v84 = v91;
      if ( *(_BYTE *)v83 != 13 )
      {
        if ( *(_BYTE *)v83 != 17 )
          goto LABEL_70;
        v85 = *(_DWORD *)(v83 + 32);
        if ( v85 <= 0x40 )
        {
          v81 = *(_QWORD *)(v83 + 24) == 1;
        }
        else
        {
          v99 = v91;
          v92 = v39;
          v86 = sub_C444A0(v83 + 24);
          v37 = v111;
          v38 = v105;
          v84 = v99;
          v39 = v92;
          v81 = v85 - 1 == v86;
        }
        if ( !v81 )
          goto LABEL_70;
      }
    }
    if ( !v81 )
      goto LABEL_70;
    goto LABEL_34;
  }
  v73 = *((_DWORD *)v72 + 8);
  if ( v73 <= 0x40 )
  {
    v58 = *((_QWORD *)v72 + 3) == 1;
  }
  else
  {
    v74 = sub_C444A0((__int64)(v72 + 24));
    v37 = v110;
    v38 = v104;
    v39 = v97;
    v58 = v73 - 1 == v74;
  }
LABEL_69:
  if ( !v58 )
  {
LABEL_70:
    v102 = v39;
    v109 = v38;
    v59 = sub_1112510(v37);
    v38 = v109;
    if ( !v102 )
    {
      if ( !v59 )
        return 0;
      v5 = 0;
      goto LABEL_36;
    }
    v107 = 0;
    v44 = v59;
LABEL_37:
    if ( v6 <= 0x21 )
      goto LABEL_38;
LABEL_54:
    if ( v6 - 34 > 7 )
      return 0;
    v53 = *(_QWORD *)(a3 - 32);
    if ( !v53 || *(_BYTE *)v53 || *(_QWORD *)(v53 + 24) != *(_QWORD *)(a3 + 80) )
      BUG();
    v54 = *(_DWORD *)(v53 + 36);
    if ( v54 == 365 )
    {
      v55 = 34;
    }
    else if ( v54 > 0x16D )
    {
      if ( v54 != 366 )
        goto LABEL_172;
      v55 = 36;
    }
    else if ( v54 == 329 )
    {
      v55 = 38;
    }
    else
    {
      if ( v54 != 330 )
        goto LABEL_172;
      v55 = 40;
    }
    v56 = sub_B53110(v6);
    if ( v5 )
    {
      if ( v56 == v55 )
      {
        v88 = sub_AD6400(*(_QWORD *)(a2 + 8));
        return sub_F162A0((__int64)a1, a2, v88);
      }
    }
    else if ( v56 != v55 )
    {
      v75 = sub_AD6450(*(_QWORD *)(a2 + 8));
      return sub_F162A0((__int64)a1, a2, v75);
    }
    goto LABEL_65;
  }
LABEL_34:
  if ( !v39 )
  {
    v5 = 1;
LABEL_36:
    v43 = v112;
    v44 = 0;
    v38 = 1;
    v107 = v5;
    v112 = v113;
    v113 = v43;
    goto LABEL_37;
  }
  v107 = 1;
  v44 = 1;
  if ( v6 > 0x21 )
    goto LABEL_54;
LABEL_38:
  if ( v6 <= 0x1F )
    return 0;
  v45 = *(_QWORD *)(a3 - 32);
  if ( v5 == (v6 == 32) )
  {
    if ( !v45 || *(_BYTE *)v45 || *(_QWORD *)(v45 + 24) != *(_QWORD *)(a3 + 80) )
      BUG();
    v76 = *(_DWORD *)(v45 + 36);
    if ( v76 == 365 )
    {
      v77 = 34;
    }
    else if ( v76 > 0x16D )
    {
      if ( v76 != 366 )
        goto LABEL_172;
      v77 = 36;
    }
    else if ( v76 == 329 )
    {
      v77 = 38;
    }
    else
    {
      if ( v76 != 330 )
        goto LABEL_172;
      v77 = 40;
    }
    v78 = sub_B531B0(v77);
    v80 = v78;
    if ( v6 == 33 )
      v80 = sub_B52870(v78);
    LOWORD(v116) = 257;
    return (unsigned __int8 *)sub_B52500(53, v80, (__int64)v112, (__int64)v113, (__int64)v115, v79, 0, 0);
  }
  if ( !v45 || *(_BYTE *)v45 || *(_QWORD *)(v45 + 24) != *(_QWORD *)(a3 + 80) )
    BUG();
  v46 = *(_DWORD *)(v45 + 36);
  if ( v46 == 365 )
  {
    v47 = 34;
    goto LABEL_47;
  }
  if ( v46 <= 0x16D )
  {
    if ( v46 == 329 )
    {
      v47 = 38;
      goto LABEL_47;
    }
    if ( v46 == 330 )
    {
      v47 = 40;
      goto LABEL_47;
    }
LABEL_172:
    BUG();
  }
  if ( v46 != 366 )
    goto LABEL_172;
  v47 = 36;
LABEL_47:
  v94 = v38;
  v48 = sub_1016CC0(v47, v112, a4, v120.m128i_i64);
  v49 = sub_1112AA0(v48);
  v50 = v94;
  if ( HIBYTE(v49) )
    goto LABEL_48;
  if ( v94 )
  {
    if ( v44 )
      goto LABEL_145;
    return 0;
  }
  if ( !v44 )
    return 0;
  v5 = v107;
  v44 = 0;
LABEL_145:
  if ( (v6 == 32) == v107 )
    return 0;
  v87 = sub_1016CC0((unsigned int)v47, v113, a4, v120.m128i_i64);
  v49 = sub_1112AA0(v87);
  if ( !HIBYTE(v49) )
    return 0;
  v107 = v5;
  v113 = v112;
LABEL_48:
  if ( (_BYTE)v49 )
  {
    v51 = v6 == 33;
LABEL_67:
    v57 = sub_AD64A0(*(_QWORD *)(a2 + 8), v51);
    return sub_F162A0((__int64)a1, a2, v57);
  }
LABEL_65:
  if ( v44 )
  {
    v51 = v107;
    goto LABEL_67;
  }
  LOWORD(v116) = 257;
  return (unsigned __int8 *)sub_B52500(53, v6, (__int64)v113, (__int64)a4, (__int64)v115, v50, 0, 0);
}
