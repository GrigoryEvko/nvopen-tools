// Function: sub_18A2540
// Address: 0x18a2540
//
__int64 __fastcall sub_18A2540(
        __int64 a1,
        __int64 a2,
        __m128 a3,
        double a4,
        double a5,
        double a6,
        double a7,
        double a8,
        double a9,
        __m128 a10)
{
  __int64 *v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  double v15; // xmm4_8
  double v16; // xmm5_8
  __int64 *v17; // rdi
  __int64 v18; // rax
  __int64 *v19; // r8
  __int64 *v20; // rcx
  __int64 *v21; // rbx
  __int64 *v22; // r13
  __int64 v23; // rsi
  __int64 *v24; // r9
  __int64 *v25; // rax
  __int64 *v26; // rcx
  __int64 **v27; // r12
  __int64 **v28; // rbx
  int v29; // r13d
  __int64 v30; // rdi
  __int64 v31; // rax
  int v32; // r15d
  __int64 *v33; // rax
  __int64 v34; // rbx
  __int64 **v35; // rbx
  __int64 **v36; // r12
  int v37; // r13d
  __int64 v38; // rdi
  char v39; // al
  __int64 v40; // r12
  int v41; // r14d
  int v42; // r13d
  int v43; // eax
  char v44; // r15
  int v45; // r14d
  __int64 v46; // r12
  unsigned __int64 v47; // rax
  unsigned __int64 v48; // rbx
  __int64 v49; // rbx
  int v50; // edx
  char v51; // r9
  __int64 v52; // r13
  __int64 v53; // r12
  char v54; // al
  unsigned __int64 v55; // r8
  __int64 v56; // r10
  _QWORD *v57; // rax
  _QWORD *v58; // r9
  __int64 v59; // rsi
  __int64 v60; // rcx
  __int64 *v61; // rax
  __int64 v62; // r9
  __int64 *v63; // r8
  unsigned __int8 v64; // al
  unsigned __int64 v65; // r12
  unsigned __int64 v66; // rax
  __int64 *v67; // rsi
  __int64 *v68; // rax
  __int64 v69; // rax
  char v70; // al
  char v71; // al
  __int64 **v72; // r12
  __int64 **v73; // rbx
  __int64 v74; // r13
  int v75; // eax
  __int64 *v76; // rsi
  __int64 v78; // [rsp+10h] [rbp-F0h]
  __int64 v79; // [rsp+18h] [rbp-E8h]
  __int64 *v80; // [rsp+20h] [rbp-E0h]
  __int64 v81; // [rsp+28h] [rbp-D8h]
  unsigned __int8 v82; // [rsp+35h] [rbp-CBh]
  unsigned __int8 v83; // [rsp+36h] [rbp-CAh]
  unsigned __int8 v84; // [rsp+37h] [rbp-C9h]
  __int64 v85; // [rsp+38h] [rbp-C8h]
  __int64 v86; // [rsp+40h] [rbp-C0h]
  __int64 v87; // [rsp+48h] [rbp-B8h]
  char v88; // [rsp+50h] [rbp-B0h]
  unsigned __int8 v89; // [rsp+50h] [rbp-B0h]
  __int64 v90; // [rsp+50h] [rbp-B0h]
  unsigned __int8 v91; // [rsp+50h] [rbp-B0h]
  char v92; // [rsp+58h] [rbp-A8h]
  __int64 v93; // [rsp+60h] [rbp-A0h] BYREF
  __int64 *v94; // [rsp+68h] [rbp-98h]
  __int64 *v95; // [rsp+70h] [rbp-90h]
  __int64 v96; // [rsp+78h] [rbp-88h]
  int v97; // [rsp+80h] [rbp-80h]
  _BYTE v98[120]; // [rsp+88h] [rbp-78h] BYREF

  v82 = 0;
  if ( (unsigned __int8)sub_384D370() )
    return v82;
  v11 = *(__int64 **)(a1 + 8);
  v12 = *v11;
  v13 = v11[1];
  if ( v12 == v13 )
LABEL_145:
    BUG();
  while ( *(_UNKNOWN **)v12 != &unk_4F98A8D )
  {
    v12 += 16;
    if ( v13 == v12 )
      goto LABEL_145;
  }
  v14 = (*(__int64 (__fastcall **)(_QWORD, void *))(**(_QWORD **)(v12 + 8) + 104LL))(*(_QWORD *)(v12 + 8), &unk_4F98A8D);
  v17 = (__int64 *)v98;
  v97 = 0;
  v18 = *(_QWORD *)(v14 + 160);
  v94 = (__int64 *)v98;
  v19 = (__int64 *)v98;
  v93 = 0;
  v79 = v18;
  v95 = (__int64 *)v98;
  v20 = *(__int64 **)(a2 + 16);
  v21 = *(__int64 **)(a2 + 24);
  v96 = 8;
  v22 = v20;
  if ( v20 == v21 )
  {
LABEL_141:
    v82 = 0;
    goto LABEL_38;
  }
  do
  {
LABEL_11:
    v23 = *v22;
    if ( v17 != v19 )
    {
LABEL_9:
      sub_16CCBA0((__int64)&v93, v23);
      v19 = v95;
      v17 = v94;
      goto LABEL_10;
    }
    v24 = &v17[HIDWORD(v96)];
    if ( v17 == v24 )
    {
LABEL_121:
      if ( HIDWORD(v96) >= (unsigned int)v96 )
        goto LABEL_9;
      ++HIDWORD(v96);
      *v24 = v23;
      v17 = v94;
      ++v93;
      v19 = v95;
    }
    else
    {
      v25 = v17;
      v26 = 0;
      while ( v23 != *v25 )
      {
        if ( *v25 == -2 )
          v26 = v25;
        if ( v24 == ++v25 )
        {
          if ( !v26 )
            goto LABEL_121;
          ++v22;
          *v26 = v23;
          v19 = v95;
          --v97;
          v17 = v94;
          ++v93;
          if ( v21 != v22 )
            goto LABEL_11;
          goto LABEL_20;
        }
      }
    }
LABEL_10:
    ++v22;
  }
  while ( v21 != v22 );
LABEL_20:
  v27 = *(__int64 ***)(a2 + 16);
  v28 = *(__int64 ***)(a2 + 24);
  if ( v27 == v28 )
    goto LABEL_141;
  v29 = 0;
  do
  {
    v30 = **v27;
    if ( v30 )
      v29 |= sub_18A22B0(v30, v79, a3, a4, a5, a6, v15, v16, a9, a10);
    ++v27;
  }
  while ( v28 != v27 );
  v82 = v29;
  v31 = *(_QWORD *)(a2 + 16);
  v78 = *(_QWORD *)(a2 + 24);
  if ( v31 == v78 )
  {
LABEL_140:
    v19 = v95;
    v17 = v94;
    goto LABEL_38;
  }
  v88 = 0;
  v32 = 0;
  v86 = v31 + 8;
  while ( 1 )
  {
    v85 = v86;
    v33 = *(__int64 **)(v86 - 8);
    v34 = *v33;
    if ( !*v33 )
      goto LABEL_32;
    if ( sub_15E4F60(*v33) || (sub_15E4B50(v34), v39) )
    {
      v88 |= sub_1560180(v34 + 112, 30) ^ 1;
      v32 |= sub_1560180(v34 + 112, 29) ^ 1;
      if ( v88 && (_BYTE)v32 )
        goto LABEL_32;
      goto LABEL_30;
    }
    if ( v88 )
    {
      if ( (_BYTE)v32 )
        goto LABEL_32;
      v40 = v34 + 112;
      LOBYTE(v41) = v88;
      v42 = sub_1560180(v34 + 112, 29);
      if ( !(_BYTE)v42 )
        goto LABEL_132;
      v88 = v42;
      goto LABEL_30;
    }
    v40 = v34 + 112;
    v41 = sub_1560180(v34 + 112, 30);
    v42 = v41 ^ 1;
    if ( (_BYTE)v32 )
    {
      if ( (_BYTE)v42 )
      {
        LOBYTE(v41) = 0;
        v92 = 0;
        v81 = v34 + 72;
        v87 = *(_QWORD *)(v34 + 80);
        v84 = 0;
        if ( v34 + 72 != v87 )
          goto LABEL_48;
        v32 = v42;
      }
      goto LABEL_30;
    }
    if ( !(unsigned __int8)sub_1560180(v34 + 112, 29) )
    {
LABEL_132:
      v92 = sub_1560180(v40, 18);
      if ( v92 )
        v92 = sub_1560180(v40, 26);
      v84 = 1;
      v81 = v34 + 72;
      v87 = *(_QWORD *)(v34 + 80);
      if ( v34 + 72 != v87 )
      {
LABEL_48:
        v43 = v32;
        v44 = v41;
        v45 = v43;
        goto LABEL_49;
      }
      goto LABEL_30;
    }
    if ( (_BYTE)v42 )
    {
      v81 = v34 + 72;
      v87 = *(_QWORD *)(v34 + 80);
      if ( v87 != v34 + 72 )
        break;
    }
LABEL_30:
    v86 += 8;
    if ( v78 == v85 )
    {
      v72 = *(__int64 ***)(a2 + 16);
      v73 = *(__int64 ***)(a2 + 24);
      if ( v72 == v73 )
        goto LABEL_140;
      while ( 1 )
      {
        v74 = **v72;
        if ( v88 || (unsigned __int8)sub_1560180(v74 + 112, 30) )
        {
          if ( !(_BYTE)v32 )
            goto LABEL_130;
        }
        else
        {
          sub_15E0D50(v74, -1, 30);
          v82 = 1;
          if ( !(_BYTE)v32 )
          {
LABEL_130:
            if ( !(unsigned __int8)sub_1560180(v74 + 112, 29) )
            {
              sub_15E0D50(v74, -1, 29);
              v82 = 1;
            }
          }
        }
        if ( v73 == ++v72 )
          goto LABEL_32;
      }
    }
  }
  v75 = v32;
  v92 = 0;
  v44 = v41;
  v84 = 0;
  v45 = v75;
LABEL_49:
  while ( 2 )
  {
    v46 = 0;
    if ( v87 )
      v46 = v87 - 24;
    v47 = sub_157EBA0(v46);
    v48 = v47;
    if ( (_BYTE)v42 && (v71 = sub_15F3330(v47)) != 0 )
    {
      v88 = v71;
    }
    else if ( v84 && *(_BYTE *)(v48 + 16) == 25 )
    {
      v45 = v84;
    }
    v49 = *(_QWORD *)(v46 + 48);
    if ( v49 == v46 + 40 )
    {
LABEL_95:
      if ( (_BYTE)v45 )
        goto LABEL_103;
      goto LABEL_96;
    }
    v50 = v42;
    v51 = v88;
    v52 = v46 + 40;
    while ( 2 )
    {
      v53 = v49 - 24;
      if ( !v49 )
        v53 = 0;
      if ( !v44 && !v51 )
      {
        if ( !(_BYTE)v50 || (v91 = v50, v70 = sub_15F3330(v53), v50 = v91, (v51 = v70) == 0) )
        {
LABEL_58:
          if ( (_BYTE)v45 == 1 || !v92 )
            goto LABEL_60;
          goto LABEL_85;
        }
LABEL_70:
        if ( *(_BYTE *)(v53 + 16) != 78 )
        {
          v51 = 1;
          goto LABEL_58;
        }
        v55 = *(_QWORD *)(v53 - 24);
        if ( *(_BYTE *)(v55 + 16) )
        {
          v51 = v92 & (v45 ^ 1);
          if ( !v51 )
          {
            v51 = 1;
            goto LABEL_60;
          }
          goto LABEL_100;
        }
        v56 = v79 + 16;
        v57 = *(_QWORD **)(v79 + 24);
        if ( v57 )
        {
          v58 = (_QWORD *)(v79 + 16);
          do
          {
            while ( 1 )
            {
              v59 = v57[2];
              v60 = v57[3];
              if ( v57[4] >= v55 )
                break;
              v57 = (_QWORD *)v57[3];
              if ( !v60 )
                goto LABEL_77;
            }
            v58 = v57;
            v57 = (_QWORD *)v57[2];
          }
          while ( v59 );
LABEL_77:
          if ( (_QWORD *)v56 != v58 && v58[4] <= v55 )
            v56 = (__int64)v58;
        }
        v61 = v94;
        v62 = *(_QWORD *)(v56 + 40);
        if ( v95 == v94 )
        {
          v63 = &v94[HIDWORD(v96)];
          if ( v94 == v63 )
          {
            v76 = v94;
          }
          else
          {
            do
            {
              if ( v62 == *v61 )
                break;
              ++v61;
            }
            while ( v63 != v61 );
            v76 = &v94[HIDWORD(v96)];
          }
        }
        else
        {
          v83 = v50;
          v90 = *(_QWORD *)(v56 + 40);
          v80 = &v95[(unsigned int)v96];
          v61 = sub_16CC9F0((__int64)&v93, v62);
          v63 = v80;
          v50 = v83;
          if ( v90 == *v61 )
          {
            if ( v95 == v94 )
              v76 = &v95[HIDWORD(v96)];
            else
              v76 = &v95[(unsigned int)v96];
          }
          else
          {
            if ( v95 != v94 )
            {
              v61 = &v95[(unsigned int)v96];
LABEL_84:
              v51 = v63 == v61;
              goto LABEL_58;
            }
            v76 = &v95[HIDWORD(v96)];
            v61 = v76;
          }
        }
        for ( ; v76 != v61; ++v61 )
        {
          if ( (unsigned __int64)*v61 < 0xFFFFFFFFFFFFFFFELL )
            break;
        }
        goto LABEL_84;
      }
      if ( v92 != 1 )
        goto LABEL_94;
      if ( !(_BYTE)v45 )
      {
        if ( v51 != 1 )
        {
          if ( (_BYTE)v50 )
          {
            v89 = v50;
            v54 = sub_15F3330(v53);
            v50 = v89;
            v51 = v54;
            if ( v54 )
              goto LABEL_70;
          }
        }
LABEL_85:
        v64 = *(_BYTE *)(v53 + 16);
        v45 = 0;
        if ( v64 <= 0x17u )
          goto LABEL_60;
        if ( v64 == 78 )
        {
LABEL_100:
          v65 = v53 | 4;
LABEL_89:
          v66 = v65 & 0xFFFFFFFFFFFFFFF8LL;
          if ( (v65 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
          {
            v45 = 0;
            goto LABEL_60;
          }
          v67 = (__int64 *)(v66 - 24);
          v68 = (__int64 *)(v66 - 72);
          if ( (v65 & 4) != 0 )
            v68 = v67;
          v45 = 0;
          v69 = *v68;
          if ( *(_BYTE *)(v69 + 16) != 20 )
            goto LABEL_60;
          v49 = *(_QWORD *)(v49 + 8);
          v45 = *(unsigned __int8 *)(v69 + 96);
          if ( v52 == v49 )
          {
LABEL_94:
            v88 = v51;
            v42 = v50;
            goto LABEL_95;
          }
        }
        else
        {
          if ( v64 == 29 )
          {
            v65 = v53 & 0xFFFFFFFFFFFFFFFBLL;
            goto LABEL_89;
          }
LABEL_60:
          v49 = *(_QWORD *)(v49 + 8);
          if ( v52 == v49 )
            goto LABEL_94;
        }
        continue;
      }
      break;
    }
    v88 = v51;
    v42 = v50;
LABEL_103:
    if ( !v88 )
    {
LABEL_96:
      v87 = *(_QWORD *)(v87 + 8);
      if ( v87 == v81 )
      {
        v32 = v45;
        goto LABEL_30;
      }
      continue;
    }
    break;
  }
LABEL_32:
  v35 = *(__int64 ***)(a2 + 16);
  v36 = *(__int64 ***)(a2 + 24);
  if ( v35 == v36 )
    goto LABEL_140;
  v37 = v82;
  do
  {
    v38 = **v35;
    if ( v38 )
      v37 |= sub_18A22B0(v38, v79, a3, a4, a5, a6, v15, v16, a9, a10);
    ++v35;
  }
  while ( v36 != v35 );
  v82 = v37;
  v19 = v95;
  v17 = v94;
LABEL_38:
  if ( v17 != v19 )
    _libc_free((unsigned __int64)v19);
  return v82;
}
