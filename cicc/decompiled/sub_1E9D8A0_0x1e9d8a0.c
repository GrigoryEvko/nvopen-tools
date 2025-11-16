// Function: sub_1E9D8A0
// Address: 0x1e9d8a0
//
__int64 __fastcall sub_1E9D8A0(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rdi
  __int64 (*v6)(); // rax
  unsigned int v7; // r15d
  __int64 v11; // rdi
  __int64 (__fastcall *v12)(__int64, __int64); // rax
  __int64 v13; // rdi
  __int64 (__fastcall *v14)(__int64, __int64); // rcx
  __int64 v15; // rax
  _BYTE *v16; // r9
  __int64 v17; // rbx
  __int64 v18; // rbx
  __int64 v19; // rdx
  _BYTE *v20; // r10
  __int64 v21; // r8
  __int64 v22; // rax
  _QWORD *v23; // rdi
  _QWORD *v24; // rax
  _QWORD *v25; // rsi
  _QWORD *v26; // r8
  int v27; // r12d
  __int64 *v28; // r9
  __int64 v29; // r15
  __int64 v30; // r13
  int v31; // eax
  __int64 v32; // r10
  _QWORD *v33; // rax
  __int64 v34; // rsi
  _BYTE *v35; // r13
  __int64 v36; // rax
  __int64 v37; // rbx
  _QWORD *v38; // r12
  int v39; // r13d
  __int64 *v40; // rax
  __int64 v41; // rcx
  __int64 v42; // r13
  __int64 *v43; // rdi
  int v44; // r14d
  __int64 v45; // r8
  _QWORD *v46; // rbx
  __int64 *v47; // rcx
  __int64 v48; // r12
  __int64 v49; // r13
  __int64 v50; // r15
  __int64 v51; // rax
  __int64 v52; // rdx
  __int64 v53; // r13
  __int64 v54; // r13
  bool v55; // al
  __int64 v56; // rax
  _QWORD *v57; // rdx
  _QWORD *v58; // rax
  _QWORD *v59; // rcx
  __int64 v60; // rax
  __int64 v61; // rdx
  __int64 v62; // rsi
  __int64 v63; // rdx
  __int64 v64; // rcx
  __int64 v65; // r8
  __int64 *v66; // rdx
  __int64 *v67; // rsi
  void *v68; // r14
  __int64 v69; // r8
  __int64 *v70; // rdx
  _BYTE *v71; // rdx
  _QWORD *v72; // rdx
  _QWORD *v73; // [rsp+8h] [rbp-1F8h]
  unsigned __int8 v74; // [rsp+8h] [rbp-1F8h]
  _QWORD *v75; // [rsp+8h] [rbp-1F8h]
  unsigned __int8 v76; // [rsp+18h] [rbp-1E8h]
  unsigned __int64 v77; // [rsp+18h] [rbp-1E8h]
  __int64 v78; // [rsp+18h] [rbp-1E8h]
  unsigned __int8 v79; // [rsp+18h] [rbp-1E8h]
  _QWORD *v80; // [rsp+18h] [rbp-1E8h]
  unsigned __int8 v81; // [rsp+20h] [rbp-1E0h]
  _QWORD *v82; // [rsp+20h] [rbp-1E0h]
  _QWORD *v83; // [rsp+20h] [rbp-1E0h]
  unsigned __int8 v84; // [rsp+20h] [rbp-1E0h]
  int v85; // [rsp+28h] [rbp-1D8h]
  unsigned __int8 v86; // [rsp+2Fh] [rbp-1D1h]
  unsigned __int64 v87; // [rsp+38h] [rbp-1C8h]
  unsigned __int64 v88; // [rsp+40h] [rbp-1C0h]
  __int64 *v90; // [rsp+48h] [rbp-1B8h]
  __int32 v91; // [rsp+48h] [rbp-1B8h]
  __int64 v92; // [rsp+48h] [rbp-1B8h]
  int v93; // [rsp+54h] [rbp-1ACh] BYREF
  int v94; // [rsp+58h] [rbp-1A8h] BYREF
  unsigned int v95; // [rsp+5Ch] [rbp-1A4h] BYREF
  __m128i v96; // [rsp+60h] [rbp-1A0h] BYREF
  __int64 v97; // [rsp+70h] [rbp-190h]
  __int64 v98; // [rsp+78h] [rbp-188h]
  __int64 v99; // [rsp+80h] [rbp-180h]
  __int64 v100; // [rsp+90h] [rbp-170h] BYREF
  _BYTE *v101; // [rsp+98h] [rbp-168h]
  _BYTE *v102; // [rsp+A0h] [rbp-160h]
  __int64 v103; // [rsp+A8h] [rbp-158h]
  int v104; // [rsp+B0h] [rbp-150h]
  _BYTE v105[40]; // [rsp+B8h] [rbp-148h] BYREF
  __int64 v106; // [rsp+E0h] [rbp-120h] BYREF
  __int64 *v107; // [rsp+E8h] [rbp-118h]
  __int64 *v108; // [rsp+F0h] [rbp-110h]
  __int64 v109; // [rsp+F8h] [rbp-108h]
  int v110; // [rsp+100h] [rbp-100h]
  _BYTE v111[40]; // [rsp+108h] [rbp-F8h] BYREF
  _BYTE *v112; // [rsp+130h] [rbp-D0h] BYREF
  __int64 v113; // [rsp+138h] [rbp-C8h]
  _BYTE v114[64]; // [rsp+140h] [rbp-C0h] BYREF
  void *src; // [rsp+180h] [rbp-80h] BYREF
  __int64 v116; // [rsp+188h] [rbp-78h]
  _BYTE v117[112]; // [rsp+190h] [rbp-70h] BYREF

  v5 = a1[29];
  v6 = *(__int64 (**)())(*(_QWORD *)v5 + 40LL);
  if ( v6 == sub_1E9BDF0 )
    return 0;
  v86 = ((__int64 (__fastcall *)(__int64, __int64, int *, int *, unsigned int *))v6)(v5, a2, &v93, &v94, &v95);
  if ( !v86 )
    return 0;
  if ( v94 > 0 )
    return 0;
  if ( v93 > 0 )
    return 0;
  v7 = sub_1E69E00(a1[31], v93);
  if ( (_BYTE)v7 )
    return 0;
  v11 = a1[30];
  v87 = *(_QWORD *)(*(_QWORD *)(a1[31] + 24LL) + 16LL * (v94 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  v12 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v11 + 112LL);
  if ( v12 != sub_1E15B90 )
    v87 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v12)(v11, v87, v95);
  if ( !v87 )
    return 0;
  v13 = a1[30];
  v14 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v13 + 112LL);
  v15 = a1[31];
  v88 = *(_QWORD *)(*(_QWORD *)(v15 + 24) + 16LL * (v93 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v14 != sub_1E15B90 )
  {
    v88 = ((__int64 (__fastcall *)(__int64, unsigned __int64, _QWORD))v14)(v13, v88, v95);
    v15 = a1[31];
  }
  v16 = v105;
  v100 = 0;
  v101 = v105;
  v102 = v105;
  v103 = 4;
  v104 = 0;
  if ( v94 < 0 )
    v17 = *(_QWORD *)(*(_QWORD *)(v15 + 24) + 16LL * (v94 & 0x7FFFFFFF) + 8);
  else
    v17 = *(_QWORD *)(*(_QWORD *)(v15 + 272) + 8LL * (unsigned int)v94);
  while ( v17 )
  {
    if ( (*(_BYTE *)(v17 + 3) & 0x10) == 0 && (*(_BYTE *)(v17 + 4) & 8) == 0 )
    {
      v19 = *(_QWORD *)(v17 + 16);
      v20 = v105;
LABEL_36:
      v21 = *(_QWORD *)(v19 + 24);
      if ( v16 == v20 )
      {
        v23 = &v16[8 * HIDWORD(v103)];
        if ( v16 == (_BYTE *)v23 )
          goto LABEL_129;
        v24 = v16;
        v25 = 0;
        do
        {
          if ( v21 == *v24 )
          {
            v22 = v19;
            goto LABEL_38;
          }
          if ( *v24 == -2 )
            v25 = v24;
          ++v24;
        }
        while ( v23 != v24 );
        if ( !v25 )
        {
LABEL_129:
          if ( HIDWORD(v103) >= (unsigned int)v103 )
            goto LABEL_37;
          ++HIDWORD(v103);
          *v23 = v21;
          v16 = v101;
          ++v100;
          v22 = *(_QWORD *)(v17 + 16);
          v20 = v102;
        }
        else
        {
          *v25 = v21;
          v20 = v102;
          --v104;
          v16 = v101;
          ++v100;
          v22 = *(_QWORD *)(v17 + 16);
        }
      }
      else
      {
LABEL_37:
        sub_16CCBA0((__int64)&v100, *(_QWORD *)(v19 + 24));
        v22 = *(_QWORD *)(v17 + 16);
        v20 = v102;
        v16 = v101;
      }
LABEL_38:
      while ( 1 )
      {
        v17 = *(_QWORD *)(v17 + 32);
        if ( !v17 )
          break;
        while ( (*(_BYTE *)(v17 + 3) & 0x10) == 0 && (*(_BYTE *)(v17 + 4) & 8) == 0 )
        {
          v19 = *(_QWORD *)(v17 + 16);
          if ( v19 != v22 )
            goto LABEL_36;
          v17 = *(_QWORD *)(v17 + 32);
          if ( !v17 )
            goto LABEL_43;
        }
      }
LABEL_43:
      v15 = a1[31];
      break;
    }
    v17 = *(_QWORD *)(v17 + 32);
  }
  v113 = 0x800000000LL;
  v116 = 0x800000000LL;
  v112 = v114;
  src = v117;
  if ( v93 < 0 )
    v18 = *(_QWORD *)(*(_QWORD *)(v15 + 24) + 16LL * (v93 & 0x7FFFFFFF) + 8);
  else
    v18 = *(_QWORD *)(*(_QWORD *)(v15 + 272) + 8LL * (unsigned int)v93);
  if ( !v18 )
    goto LABEL_30;
  if ( (*(_BYTE *)(v18 + 3) & 0x10) != 0 || (*(_BYTE *)(v18 + 4) & 8) != 0 )
  {
    v18 = *(_QWORD *)(v18 + 32);
    if ( !v18 )
      goto LABEL_30;
    while ( (*(_BYTE *)(v18 + 3) & 0x10) != 0 || (*(_BYTE *)(v18 + 4) & 8) != 0 )
    {
      v18 = *(_QWORD *)(v18 + 32);
      if ( !v18 )
        goto LABEL_28;
    }
  }
  v26 = a1;
  v27 = v86;
  LODWORD(v28) = v7;
  v29 = a4;
LABEL_54:
  v30 = *(_QWORD *)(v18 + 16);
  if ( a2 == v30 )
    goto LABEL_69;
  v31 = **(unsigned __int16 **)(v30 + 16);
  if ( v31 == 45 || !**(_WORD **)(v30 + 16) )
  {
    v27 = 0;
    goto LABEL_69;
  }
  if ( v88 && v95 != ((*(_DWORD *)v18 >> 8) & 0xFFF) || v31 == 10 )
    goto LABEL_69;
  v32 = *(_QWORD *)(v30 + 24);
  if ( a3 == v32 )
  {
    v57 = *(_QWORD **)(v29 + 16);
    v58 = *(_QWORD **)(v29 + 8);
    if ( v57 == v58 )
    {
      v59 = &v58[*(unsigned int *)(v29 + 28)];
      if ( v58 == v59 )
      {
        v72 = *(_QWORD **)(v29 + 8);
      }
      else
      {
        do
        {
          if ( v30 == *v58 )
            break;
          ++v58;
        }
        while ( v59 != v58 );
        v72 = v59;
      }
    }
    else
    {
      v75 = v26;
      v79 = (unsigned __int8)v28;
      v83 = &v57[*(unsigned int *)(v29 + 24)];
      v58 = sub_16CC9F0(v29, *(_QWORD *)(v18 + 16));
      v59 = v83;
      LODWORD(v28) = v79;
      v26 = v75;
      if ( v30 == *v58 )
      {
        v61 = *(_QWORD *)(v29 + 16);
        if ( v61 == *(_QWORD *)(v29 + 8) )
          v62 = *(unsigned int *)(v29 + 28);
        else
          v62 = *(unsigned int *)(v29 + 24);
        v72 = (_QWORD *)(v61 + 8 * v62);
      }
      else
      {
        v60 = *(_QWORD *)(v29 + 16);
        if ( v60 != *(_QWORD *)(v29 + 8) )
        {
          v58 = (_QWORD *)(v60 + 8LL * *(unsigned int *)(v29 + 24));
          goto LABEL_123;
        }
        v72 = (_QWORD *)(v60 + 8LL * *(unsigned int *)(v29 + 28));
        v58 = v72;
      }
    }
    while ( v72 != v58 && *v58 >= 0xFFFFFFFFFFFFFFFELL )
      ++v58;
LABEL_123:
    if ( v59 == v58 )
      goto LABEL_66;
LABEL_69:
    while ( 1 )
    {
      v18 = *(_QWORD *)(v18 + 32);
      if ( !v18 )
        break;
      while ( (*(_BYTE *)(v18 + 3) & 0x10) == 0 )
      {
        if ( (*(_BYTE *)(v18 + 4) & 8) == 0 )
          goto LABEL_54;
        v18 = *(_QWORD *)(v18 + 32);
        if ( !v18 )
          goto LABEL_73;
      }
    }
LABEL_73:
    v7 = (unsigned int)v28;
    LODWORD(v28) = v27;
    v37 = (unsigned int)v113;
    v38 = v26;
    if ( (_BYTE)v28 )
    {
      v39 = v116;
      if ( (_DWORD)v116 )
      {
        v68 = src;
        v69 = 8LL * (unsigned int)v116;
        if ( (unsigned int)v116 > HIDWORD(v113) - (unsigned __int64)(unsigned int)v113 )
        {
          v92 = 8LL * (unsigned int)v116;
          sub_16CD150((__int64)&v112, v114, (unsigned int)v116 + (unsigned __int64)(unsigned int)v113, 8, v69, (int)v28);
          v37 = (unsigned int)v113;
          v69 = v92;
        }
        memcpy(&v112[8 * v37], v68, v69);
        LODWORD(v113) = v113 + v39;
        LODWORD(v37) = v113;
      }
    }
    goto LABEL_75;
  }
  v33 = v101;
  if ( v102 == v101 )
  {
    v35 = &v101[8 * HIDWORD(v103)];
    if ( v101 == v35 )
    {
      v71 = v101;
    }
    else
    {
      do
      {
        if ( v32 == *v33 )
          break;
        ++v33;
      }
      while ( v35 != (_BYTE *)v33 );
      v71 = &v101[8 * HIDWORD(v103)];
    }
  }
  else
  {
    v34 = *(_QWORD *)(v30 + 24);
    v73 = v26;
    v76 = (unsigned __int8)v28;
    v35 = &v102[8 * (unsigned int)v103];
    v33 = sub_16CC9F0((__int64)&v100, v34);
    v32 = v34;
    LODWORD(v28) = v76;
    v26 = v73;
    if ( v34 == *v33 )
    {
      if ( v102 == v101 )
        v71 = &v102[8 * HIDWORD(v103)];
      else
        v71 = &v102[8 * (unsigned int)v103];
    }
    else
    {
      if ( v102 != v101 )
      {
        v33 = &v102[8 * (unsigned int)v103];
        goto LABEL_65;
      }
      v33 = &v102[8 * HIDWORD(v103)];
      v71 = v33;
    }
  }
  while ( v71 != (_BYTE *)v33 && *v33 >= 0xFFFFFFFFFFFFFFFELL )
    ++v33;
LABEL_65:
  if ( v35 != (_BYTE *)v33 )
  {
LABEL_66:
    v36 = (unsigned int)v113;
    if ( (unsigned int)v113 >= HIDWORD(v113) )
    {
      v80 = v26;
      v84 = (unsigned __int8)v28;
      sub_16CD150((__int64)&v112, v114, 0, 8, (int)v26, (int)v28);
      v36 = (unsigned int)v113;
      v26 = v80;
      LODWORD(v28) = v84;
    }
    *(_QWORD *)&v112[8 * v36] = v18;
    LODWORD(v113) = v113 + 1;
    goto LABEL_69;
  }
  if ( byte_4FC89E0 )
  {
    v54 = v26[32];
    v74 = (unsigned __int8)v28;
    v82 = v26;
    v78 = v32;
    sub_1E06620(v54);
    v55 = sub_1E05550(*(_QWORD *)(v54 + 1312), a3, v78);
    v26 = v82;
    LODWORD(v28) = v74;
    if ( v55 )
    {
      v56 = (unsigned int)v116;
      if ( (unsigned int)v116 >= HIDWORD(v116) )
      {
        sub_16CD150((__int64)&src, v117, 0, 8, (int)v82, v74);
        v56 = (unsigned int)v116;
        v26 = v82;
        LODWORD(v28) = v74;
      }
      *((_QWORD *)src + v56) = v18;
      LODWORD(v116) = v116 + 1;
      goto LABEL_69;
    }
  }
  LODWORD(v37) = v113;
  v7 = (unsigned int)v28;
  v38 = v26;
LABEL_75:
  if ( !(_DWORD)v37 )
    goto LABEL_28;
  v40 = (__int64 *)v111;
  v106 = 0;
  v107 = (__int64 *)v111;
  v41 = v38[31];
  v108 = (__int64 *)v111;
  v109 = 4;
  v110 = 0;
  if ( v94 < 0 )
    v42 = *(_QWORD *)(*(_QWORD *)(v41 + 24) + 16LL * (v94 & 0x7FFFFFFF) + 8);
  else
    v42 = *(_QWORD *)(*(_QWORD *)(v41 + 272) + 8LL * (unsigned int)v94);
  while ( 1 )
  {
    if ( !v42 )
    {
      v43 = (__int64 *)v111;
      v77 = *(_QWORD *)(*(_QWORD *)(v41 + 24) + 16LL * (v93 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
LABEL_83:
      v85 = v37;
      v44 = 0;
      v45 = v7;
      v46 = v38;
      while ( 1 )
      {
        v48 = *(_QWORD *)&v112[8 * v44];
        v49 = *(_QWORD *)(v48 + 16);
        v50 = *(_QWORD *)(v49 + 24);
        if ( v43 == v40 )
        {
          v47 = &v40[HIDWORD(v109)];
          if ( v47 == v40 )
          {
            v70 = v40;
          }
          else
          {
            do
            {
              if ( v50 == *v40 )
                break;
              ++v40;
            }
            while ( v47 != v40 );
            v70 = v47;
          }
          goto LABEL_97;
        }
        v81 = v45;
        v90 = &v43[(unsigned int)v109];
        v40 = sub_16CC9F0((__int64)&v106, *(_QWORD *)(v49 + 24));
        v47 = v90;
        v45 = v81;
        if ( v50 == *v40 )
          break;
        if ( v108 == v107 )
        {
          v70 = &v108[HIDWORD(v109)];
          v40 = v70;
          goto LABEL_97;
        }
        v40 = &v108[(unsigned int)v109];
LABEL_87:
        if ( v47 == v40 )
        {
          if ( !(_BYTE)v45 )
          {
            sub_1E69E80(v46[31], v94);
            sub_1E69410((__int64 *)v46[31], v94, v87, 0);
          }
          v91 = sub_1E6B9A0(v46[31], v77, (unsigned __int8 *)byte_3F871B3, 0, v45, (int)v28);
          v51 = sub_1E9D330(v50, v49, (__int64 *)(v49 + 64), *(_QWORD *)(v46[29] + 8LL) + 960LL, v91);
          v53 = v52;
          v97 = 0;
          v96.m128i_i32[2] = v94;
          v98 = 0;
          v99 = 0;
          v96.m128i_i64[0] = (unsigned __int16)(v95 & 0xFFF) << 8;
          sub_1E1A9C0(v52, v51, &v96);
          if ( v88 )
          {
            **(_DWORD **)(v53 + 32) = ((v95 & 0xFFF) << 8) | **(_DWORD **)(v53 + 32) & 0xFFF000FF;
            *(_BYTE *)(*(_QWORD *)(v53 + 32) + 4LL) |= 1u;
          }
          sub_1E310D0(v48, v91);
          v45 = v86;
        }
        v43 = v108;
        v40 = v107;
        if ( ++v44 == v85 )
        {
          v7 = v45;
          goto LABEL_157;
        }
      }
      if ( v108 == v107 )
        v70 = &v108[HIDWORD(v109)];
      else
        v70 = &v108[(unsigned int)v109];
LABEL_97:
      while ( v70 != v40 && (unsigned __int64)*v40 >= 0xFFFFFFFFFFFFFFFELL )
        ++v40;
      goto LABEL_87;
    }
    if ( (*(_BYTE *)(v42 + 3) & 0x10) == 0 && (*(_BYTE *)(v42 + 4) & 8) == 0 )
      break;
    v42 = *(_QWORD *)(v42 + 32);
  }
  v63 = *(_QWORD *)(v42 + 16);
  v43 = (__int64 *)v111;
LABEL_146:
  if ( **(_WORD **)(v63 + 16) == 45 || (v64 = v63, !**(_WORD **)(v63 + 16)) )
  {
    v65 = *(_QWORD *)(v63 + 24);
    if ( v43 != v40 )
      goto LABEL_149;
    v28 = &v43[HIDWORD(v109)];
    if ( v43 == v28 )
    {
LABEL_174:
      if ( HIDWORD(v109) >= (unsigned int)v109 )
      {
LABEL_149:
        sub_16CCBA0((__int64)&v106, v65);
        v43 = v108;
        v40 = v107;
      }
      else
      {
        ++HIDWORD(v109);
        *v28 = v65;
        v40 = v107;
        ++v106;
        v43 = v108;
      }
    }
    else
    {
      v66 = v43;
      v67 = 0;
      while ( v65 != *v66 )
      {
        if ( *v66 == -2 )
          v67 = v66;
        if ( v28 == ++v66 )
        {
          if ( !v67 )
            goto LABEL_174;
          *v67 = v65;
          v43 = v108;
          --v110;
          v40 = v107;
          ++v106;
          break;
        }
      }
    }
    v64 = *(_QWORD *)(v42 + 16);
  }
  while ( 1 )
  {
    v42 = *(_QWORD *)(v42 + 32);
    if ( !v42 )
      break;
    while ( (*(_BYTE *)(v42 + 3) & 0x10) == 0 && (*(_BYTE *)(v42 + 4) & 8) == 0 )
    {
      v63 = *(_QWORD *)(v42 + 16);
      if ( v63 != v64 )
        goto LABEL_146;
      v42 = *(_QWORD *)(v42 + 32);
      if ( !v42 )
        goto LABEL_156;
    }
  }
LABEL_156:
  LODWORD(v37) = v113;
  v77 = *(_QWORD *)(*(_QWORD *)(v38[31] + 24LL) + 16LL * (v93 & 0x7FFFFFFF)) & 0xFFFFFFFFFFFFFFF8LL;
  if ( (_DWORD)v113 )
    goto LABEL_83;
LABEL_157:
  if ( v40 != v43 )
    _libc_free((unsigned __int64)v43);
LABEL_28:
  if ( src != v117 )
    _libc_free((unsigned __int64)src);
LABEL_30:
  if ( v112 != v114 )
    _libc_free((unsigned __int64)v112);
  if ( v102 != v101 )
    _libc_free((unsigned __int64)v102);
  return v7;
}
