// Function: sub_1E1FC80
// Address: 0x1e1fc80
//
__int64 __fastcall sub_1E1FC80(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rbx
  __int64 v4; // r12
  unsigned __int64 v5; // rax
  __int64 v6; // r13
  int v8; // edx
  int v9; // ecx
  __int64 v10; // rdi
  unsigned int v11; // edx
  __int64 v12; // rsi
  __int64 v13; // r14
  __int64 v14; // r15
  __int64 v15; // rdx
  __int64 v16; // rsi
  __int64 (*v17)(); // rax
  __int64 v18; // rdi
  __int64 (__fastcall *v19)(__int64, __int64); // rdx
  __int16 v20; // dx
  __int64 v21; // rcx
  int v22; // r9d
  __int64 v23; // rax
  __int64 v24; // r14
  __int64 v25; // r15
  __int64 (*v26)(); // rax
  int v27; // eax
  __int64 v28; // r14
  int v29; // r11d
  unsigned int v30; // r8d
  __int64 v31; // rax
  int v32; // ebx
  __int64 v33; // r13
  __int64 v34; // rcx
  __int64 v35; // r15
  __int16 v36; // ax
  _BOOL4 v37; // eax
  __int64 v38; // rcx
  __int64 v39; // r13
  int v40; // r12d
  unsigned int v41; // r14d
  __int64 v42; // rbx
  __int64 v43; // rax
  __int64 v44; // rdi
  __int64 (*v45)(); // rax
  int v46; // r8d
  unsigned int *v47; // r11
  int v48; // r15d
  char v49; // cl
  __int64 v50; // rsi
  unsigned int v51; // r14d
  int v52; // ebx
  __int64 v53; // rax
  __int64 v54; // rax
  char v55; // al
  __int64 v56; // rax
  __int64 v57; // r13
  __int64 v58; // r15
  __int64 v59; // r14
  __int64 v60; // rax
  __int16 v61; // dx
  __int64 v62; // rax
  unsigned int *v63; // rsi
  unsigned int *v64; // rdx
  unsigned int *v65; // rax
  __int64 v66; // r8
  __int64 v67; // rdx
  __int64 v68; // rcx
  int v69; // r10d
  __int64 v70; // r9
  __int64 *v71; // rdx
  __int64 *v72; // rdi
  int v73; // eax
  __int64 v74; // rax
  __int64 v75; // r14
  __int64 v76; // r15
  __int64 (*v77)(); // rax
  __int64 v78; // rdx
  __int64 v79; // rsi
  int v80; // r8d
  unsigned int v81; // ecx
  int *v82; // rax
  int v83; // r11d
  int v84; // eax
  int v85; // r10d
  int v86; // [rsp+0h] [rbp-80h]
  __int64 v87; // [rsp+0h] [rbp-80h]
  __int64 v88; // [rsp+8h] [rbp-78h]
  __int64 v89; // [rsp+10h] [rbp-70h]
  unsigned int v90; // [rsp+18h] [rbp-68h]
  int v91; // [rsp+18h] [rbp-68h]
  unsigned __int8 v92; // [rsp+1Eh] [rbp-62h]
  char v93; // [rsp+1Fh] [rbp-61h]
  _QWORD *v94; // [rsp+20h] [rbp-60h]
  unsigned __int8 v95; // [rsp+20h] [rbp-60h]
  char v96; // [rsp+28h] [rbp-58h]
  __int64 v97; // [rsp+28h] [rbp-58h]
  char v98[8]; // [rsp+30h] [rbp-50h] BYREF
  unsigned int *v99; // [rsp+38h] [rbp-48h]
  int v100; // [rsp+40h] [rbp-40h]
  int v101; // [rsp+48h] [rbp-38h]

  v2 = *(_QWORD *)(a2 + 16);
  if ( *(_WORD *)v2 == 9 )
    goto LABEL_9;
  v3 = a1;
  v4 = a2;
  if ( byte_4FC6740 && *(_WORD *)v2 == 15 )
  {
    v13 = *(_QWORD *)(a1 + 248);
    v14 = *(_QWORD *)(a1 + 264);
    v15 = sub_1E15F70(a2);
    v16 = *(unsigned int *)(*(_QWORD *)(a2 + 32) + 48LL);
    if ( (int)v16 < 0
      || (v17 = *(__int64 (**)())(*(_QWORD *)v13 + 80LL), v17 == sub_1E1C7F0)
      || !((unsigned __int8 (__fastcall *)(__int64, __int64, __int64))v17)(v13, v16, v15)
      || ((v56 = *(unsigned int *)(*(_QWORD *)(v4 + 32) + 8LL), (int)v56 < 0)
        ? (v57 = *(_QWORD *)(*(_QWORD *)(v14 + 24) + 16 * (v56 & 0x7FFFFFFF) + 8))
        : (v57 = *(_QWORD *)(*(_QWORD *)(v14 + 272) + 8 * v56)),
          !v57) )
    {
LABEL_12:
      v2 = *(_QWORD *)(v4 + 16);
      goto LABEL_4;
    }
    while ( (*(_BYTE *)(v57 + 3) & 0x10) != 0 )
    {
      v57 = *(_QWORD *)(v57 + 32);
      if ( !v57 )
        goto LABEL_12;
    }
    v97 = v14;
    v58 = v13;
    v59 = *(_QWORD *)(v57 + 16);
LABEL_95:
    v60 = *(_QWORD *)(v59 + 16);
    if ( *(_WORD *)v60 != 1 || (*(_BYTE *)(*(_QWORD *)(v59 + 32) + 64LL) & 0x10) == 0 )
    {
      v61 = *(_WORD *)(v59 + 46);
      if ( (v61 & 4) != 0 || (v61 & 8) == 0 )
      {
        if ( (*(_QWORD *)(v60 + 8) & 0x20000LL) == 0 )
          goto LABEL_100;
      }
      else if ( !sub_1E15D00(v59, 0x20000u, 1) )
      {
        goto LABEL_100;
      }
    }
    if ( (unsigned __int8)sub_1E1CC30(v59, v58, v97) )
      goto LABEL_9;
LABEL_100:
    v62 = *(_QWORD *)(v57 + 16);
    while ( 1 )
    {
      v57 = *(_QWORD *)(v57 + 32);
      if ( !v57 )
        goto LABEL_12;
      if ( (*(_BYTE *)(v57 + 3) & 0x10) == 0 )
      {
        v59 = *(_QWORD *)(v57 + 16);
        if ( v62 != v59 )
          goto LABEL_95;
      }
    }
  }
LABEL_4:
  v5 = *(_QWORD *)(v2 + 8);
  v6 = (v5 >> 9) & 1;
  if ( ((v5 >> 9) & 1) != 0 )
  {
LABEL_5:
    LODWORD(v6) = 0;
    return (unsigned int)v6;
  }
  v8 = *(_DWORD *)(a1 + 736);
  if ( v8 )
  {
    v9 = v8 - 1;
    v10 = *(_QWORD *)(a1 + 720);
    v11 = (v8 - 1) & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v12 = *(_QWORD *)(v10 + 8LL * v11);
    if ( v4 == v12 )
      goto LABEL_9;
    v46 = 1;
    while ( v12 != -8 )
    {
      v11 = v9 & (v46 + v11);
      v12 = *(_QWORD *)(v10 + 8LL * v11);
      if ( v4 == v12 )
        goto LABEL_9;
      ++v46;
    }
  }
  v18 = *(_QWORD *)(v3 + 232);
  v19 = *(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v18 + 136LL);
  if ( v19 == sub_1DF74E0 )
  {
    v20 = *(_WORD *)(v4 + 46);
    if ( (v20 & 4) == 0 && (v20 & 8) != 0 )
      v96 = sub_1E15D00(v4, 0x4000000u, 2);
    else
      v96 = (v5 & 0x4000000) != 0;
  }
  else
  {
    v96 = v19(v18, v4);
  }
  if ( v96 || (v21 = *(_QWORD *)(v4 + 16), (v96 = *(_WORD *)v21 == 15 || *(_WORD *)v21 == 10) != 0) )
  {
LABEL_19:
    v93 = sub_1E1E9F0(v3, v4);
    if ( v93 && v96 )
      goto LABEL_5;
  }
  else
  {
    v48 = *(unsigned __int8 *)(v21 + 4);
    v49 = v48 == 0 || *(_DWORD *)(v4 + 40) == 0;
    if ( v49 )
    {
      v93 = sub_1E1E9F0(v3, v4);
    }
    else
    {
      v95 = v6;
      v50 = v3 + 272;
      v6 = v3;
      v51 = 0;
      v52 = *(_DWORD *)(v4 + 40);
      while ( 1 )
      {
        v53 = *(_QWORD *)(v4 + 32) + 40LL * v51;
        if ( !*(_BYTE *)v53 && (*(_BYTE *)(v53 + 3) & 0x10) != 0 )
        {
          --v48;
          if ( *(int *)(v53 + 8) <= 0 )
          {
            v49 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, _QWORD))(**(_QWORD **)(v6 + 232) + 888LL))(
                    *(_QWORD *)(v6 + 232),
                    v50,
                    v4,
                    v51);
            if ( !v49 )
              break;
          }
        }
        ++v51;
        if ( !v48 || v52 == v51 )
        {
          v3 = v6;
          v96 = v49;
          LODWORD(v6) = v95;
          goto LABEL_19;
        }
      }
      v3 = v6;
      LODWORD(v6) = v95;
      v96 = 0;
      v93 = sub_1E1E9F0(v3, v4);
    }
  }
  v23 = *(_QWORD *)(v4 + 16);
  if ( *(_WORD *)v23 == 9 )
    goto LABEL_9;
  if ( (*(_BYTE *)(v23 + 11) & 2) != 0 )
  {
    v24 = *(_QWORD *)(v3 + 232);
    v25 = *(_QWORD *)(v3 + 576);
    v26 = *(__int64 (**)())(*(_QWORD *)v24 + 16LL);
    if ( v26 != sub_1E1C800
      && ((unsigned __int8 (__fastcall *)(_QWORD, __int64, _QWORD))v26)(
           *(_QWORD *)(v3 + 232),
           v4,
           *(_QWORD *)(v3 + 576))
      || (unsigned __int8)sub_1F3B9C0(v24, v4, v25) )
    {
      goto LABEL_9;
    }
    v23 = *(_QWORD *)(v4 + 16);
  }
  v27 = *(unsigned __int16 *)(v23 + 2);
  v28 = 0;
  v88 = v3 + 272;
  if ( !v27 )
  {
LABEL_67:
    sub_1E1F460((__int64)v98, v3, v4, 0, 0, v22);
    v47 = v99;
    if ( !v100 )
      goto LABEL_68;
    v63 = &v99[2 * v101];
    if ( v99 == v63 )
      goto LABEL_68;
    v64 = v99;
    while ( 1 )
    {
      v65 = v64;
      if ( *v64 <= 0xFFFFFFFD )
        break;
      v64 += 2;
      if ( v63 == v64 )
        goto LABEL_68;
    }
    if ( v64 == v63 )
      goto LABEL_68;
    while ( 1 )
    {
      v66 = v65[1];
      if ( (int)v66 > 0 )
      {
        v67 = *v65;
        v68 = *(_QWORD *)(v3 + 984);
        v69 = *(_DWORD *)(v68 + 4 * v67);
        v70 = 4 * v67;
        if ( v96 && !byte_4FC6900 )
          goto LABEL_124;
        v71 = *(__int64 **)(v3 + 1032);
        v72 = &v71[6 * *(unsigned int *)(v3 + 1040)];
        if ( v71 != v72 )
          break;
      }
LABEL_135:
      v65 += 2;
      if ( v65 != v63 )
      {
        while ( *v65 > 0xFFFFFFFD )
        {
          v65 += 2;
          if ( v63 == v65 )
            goto LABEL_68;
        }
        if ( v65 != v63 )
          continue;
      }
      goto LABEL_68;
    }
    while ( 1 )
    {
      v68 = *v71;
      if ( v69 <= (int)v66 + *(_DWORD *)(*v71 + v70) )
        break;
      v71 += 6;
      if ( v72 == v71 )
        goto LABEL_135;
    }
LABEL_124:
    if ( v93 )
      goto LABEL_69;
    if ( byte_4FC69E0 )
    {
      v73 = *(_DWORD *)(v3 + 1848);
      if ( v73 == 2 )
      {
        if ( sub_1E1C9C0(v3, *(_QWORD *)(v4 + 24)) )
          goto LABEL_128;
      }
      else if ( !v73 )
      {
        goto LABEL_128;
      }
      v78 = *(unsigned int *)(v3 + 1840);
      if ( (_DWORD)v78 )
      {
        v79 = *(_QWORD *)(v3 + 1824);
        v80 = **(unsigned __int16 **)(v4 + 16);
        v81 = (v78 - 1) & (37 * v80);
        v82 = (int *)(v79 + 32LL * v81);
        v83 = *v82;
        if ( v80 == *v82 )
        {
LABEL_149:
          if ( v82 != (int *)(v79 + 32 * v78) && **(_WORD **)(v4 + 16) != 9 && sub_1E1C890(v3, v4, (_QWORD *)v82 + 1) )
            goto LABEL_128;
        }
        else
        {
          v84 = 1;
          while ( v83 != -1 )
          {
            v85 = v84 + 1;
            v81 = (v78 - 1) & (v84 + v81);
            v82 = (int *)(v79 + 32LL * v81);
            v83 = *v82;
            if ( v80 == *v82 )
              goto LABEL_149;
            v84 = v85;
          }
        }
      }
      v47 = v99;
      LODWORD(v6) = 0;
      goto LABEL_69;
    }
LABEL_128:
    v74 = *(_QWORD *)(v4 + 16);
    if ( *(_WORD *)v74 != 9 )
    {
      v75 = *(_QWORD *)(v3 + 576);
      if ( (*(_BYTE *)(v74 + 11) & 2) != 0 )
      {
        v76 = *(_QWORD *)(v3 + 232);
        v77 = *(__int64 (**)())(*(_QWORD *)v76 + 16LL);
        if ( v77 != sub_1E1C800 )
        {
          LODWORD(v6) = ((__int64 (__fastcall *)(_QWORD, __int64, _QWORD, __int64, __int64, __int64))v77)(
                          *(_QWORD *)(v3 + 232),
                          v4,
                          *(_QWORD *)(v3 + 576),
                          v68,
                          v66,
                          v70);
          if ( (_BYTE)v6 )
            goto LABEL_134;
        }
        LODWORD(v6) = sub_1F3B9C0(v76, v4, v75);
        if ( (_BYTE)v6 )
          goto LABEL_134;
        v75 = *(_QWORD *)(v3 + 576);
      }
      LODWORD(v6) = sub_1E176D0(v4, v75);
LABEL_134:
      v47 = v99;
      goto LABEL_69;
    }
    v47 = v99;
LABEL_68:
    LODWORD(v6) = 1;
LABEL_69:
    j___libc_free_0(v47);
    return (unsigned int)v6;
  }
  v92 = v6;
  v29 = v27;
  v94 = (_QWORD *)v3;
  while ( 1 )
  {
    v30 = v28;
    v31 = *(_QWORD *)(v4 + 32) + 40 * v28;
    if ( *(_BYTE *)v31 )
      goto LABEL_28;
    if ( (*(_BYTE *)(v31 + 3) & 0x20) != 0 )
      goto LABEL_28;
    v32 = *(_DWORD *)(v31 + 8);
    if ( v32 >= 0 )
      goto LABEL_28;
    if ( (*(_BYTE *)(v31 + 3) & 0x10) == 0 )
      goto LABEL_28;
    v33 = *(_QWORD *)(*(_QWORD *)(v94[33] + 24LL) + 16LL * (v32 & 0x7FFFFFFF) + 8);
    if ( !v33 )
      goto LABEL_28;
    if ( (*(_BYTE *)(v33 + 3) & 0x10) == 0 && (*(_BYTE *)(v33 + 4) & 8) == 0 )
      goto LABEL_36;
    v34 = *(_QWORD *)(v33 + 32);
    v54 = v34;
    if ( !v34 )
      goto LABEL_28;
    while ( (*(_BYTE *)(v54 + 3) & 0x10) != 0 || (*(_BYTE *)(v54 + 4) & 8) != 0 )
    {
      v54 = *(_QWORD *)(v54 + 32);
      if ( !v54 )
        goto LABEL_28;
    }
    if ( (*(_BYTE *)(v33 + 3) & 0x10) == 0 )
    {
LABEL_36:
      if ( (*(_BYTE *)(v33 + 4) & 8) == 0 )
        goto LABEL_43;
      v34 = *(_QWORD *)(v33 + 32);
    }
    v33 = v34;
    if ( !v34 )
      goto LABEL_28;
    while ( (*(_BYTE *)(v33 + 3) & 0x10) != 0 || (*(_BYTE *)(v33 + 4) & 8) != 0 )
    {
      v33 = *(_QWORD *)(v33 + 32);
      if ( !v33 )
        goto LABEL_28;
    }
LABEL_43:
    v35 = *(_QWORD *)(v33 + 16);
LABEL_44:
    v36 = **(_WORD **)(v35 + 16);
    if ( v36 == 15 || v36 == 10 )
    {
LABEL_61:
      while ( 1 )
      {
        v33 = *(_QWORD *)(v33 + 32);
        if ( !v33 )
          goto LABEL_28;
        if ( (*(_BYTE *)(v33 + 3) & 0x10) == 0 && (*(_BYTE *)(v33 + 4) & 8) == 0 && *(_QWORD *)(v33 + 16) != v35 )
        {
          v35 = *(_QWORD *)(v33 + 16);
          goto LABEL_44;
        }
      }
    }
    v90 = v30;
    v86 = v29;
    v37 = sub_1DA1810(v94[76] + 56LL, *(_QWORD *)(v35 + 24));
    v29 = v86;
    v30 = v90;
    if ( !v37 )
    {
      v35 = *(_QWORD *)(v33 + 16);
      goto LABEL_61;
    }
    if ( *(_DWORD *)(v35 + 40) )
      break;
LABEL_28:
    if ( v29 == (_DWORD)++v28 )
    {
      LODWORD(v6) = v92;
      v3 = (__int64)v94;
      goto LABEL_67;
    }
  }
  v89 = v28;
  v38 = v4;
  v39 = 0;
  v40 = v32;
  v41 = v90;
  v42 = *(unsigned int *)(v35 + 40);
  while ( 1 )
  {
    v43 = *(_QWORD *)(v35 + 32) + 40 * v39;
    if ( !*(_BYTE *)v43 && (*(_BYTE *)(v43 + 3) & 0x10) == 0 && v40 == *(_DWORD *)(v43 + 8) )
    {
      v44 = v94[29];
      v45 = *(__int64 (**)())(*(_QWORD *)v44 + 880LL);
      if ( v45 != sub_1E1C880 )
      {
        v91 = v29;
        v87 = v38;
        v55 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD, __int64, _QWORD, __int64, __int64))v45)(
                v44,
                v88,
                v94[33],
                v38,
                v41,
                v35,
                v39);
        v29 = v91;
        v38 = v87;
        if ( v55 )
          break;
      }
    }
    if ( ++v39 == v42 )
    {
      v28 = v89;
      v4 = v38;
      goto LABEL_28;
    }
  }
LABEL_9:
  LODWORD(v6) = 1;
  return (unsigned int)v6;
}
