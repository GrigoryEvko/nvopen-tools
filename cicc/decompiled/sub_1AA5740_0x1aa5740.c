// Function: sub_1AA5740
// Address: 0x1aa5740
//
void __fastcall sub_1AA5740(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, char a6)
{
  __int64 *v6; // r14
  __int64 *v8; // rbx
  _BYTE *v9; // r9
  unsigned __int64 v10; // r8
  __int64 v11; // rsi
  __int64 *v12; // rdi
  _QWORD *v13; // rax
  __int64 i; // r15
  __int64 v15; // rbx
  char v16; // di
  __int64 v17; // r9
  __int64 v18; // rax
  char v19; // si
  unsigned int v20; // r12d
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // r13
  unsigned __int64 v25; // rsi
  __int64 v26; // rbx
  __int64 j; // r12
  __int64 v28; // rdx
  __int64 v29; // r14
  _QWORD *v30; // rax
  _QWORD *v31; // r15
  char v32; // al
  __int64 v33; // rax
  __int64 v34; // rdx
  __int64 v35; // r12
  __int64 v36; // rax
  __int64 v37; // r9
  __int64 v38; // r13
  int v39; // eax
  __int64 v40; // r14
  __int64 v41; // r15
  __int64 v42; // rcx
  __int64 v43; // r13
  __int64 v44; // rsi
  __int64 v45; // r12
  _QWORD *v46; // rax
  __int64 v47; // rdx
  __int64 v48; // rdx
  int v49; // eax
  __int64 v50; // rax
  int v51; // esi
  __int64 v52; // rsi
  __int64 *v53; // rax
  __int64 v54; // rdi
  unsigned __int64 v55; // rsi
  __int64 v56; // rsi
  __int64 v57; // rdx
  __int64 v58; // rax
  bool v59; // cf
  __int64 v60; // rdx
  __int64 v61; // rax
  __int64 v62; // r13
  __int64 v63; // r14
  __int64 v64; // rsi
  __int64 v65; // r15
  _QWORD *v66; // rax
  _BYTE *v67; // r12
  __int64 v68; // rdx
  __int64 v69; // r13
  __int64 v70; // r15
  int v71; // eax
  __int64 v72; // rax
  int v73; // edx
  __int64 v74; // rdx
  __int64 *v75; // rax
  __int64 v76; // rcx
  unsigned __int64 v77; // rdx
  __int64 v78; // rdx
  __int64 v79; // rcx
  __int64 v80; // rdx
  __int64 v81; // rdx
  __int64 v82; // rbx
  int v83; // eax
  __int64 v84; // rax
  int v85; // edx
  __int64 v86; // rdx
  __int64 v87; // rcx
  unsigned __int64 v88; // rdx
  __int64 v89; // rcx
  _QWORD *v90; // rdx
  int v91; // [rsp+8h] [rbp-158h]
  __int64 v96; // [rsp+30h] [rbp-130h]
  __int64 v97; // [rsp+38h] [rbp-128h]
  __int64 v98; // [rsp+40h] [rbp-120h]
  _BYTE *v99; // [rsp+40h] [rbp-120h]
  __int64 v100; // [rsp+40h] [rbp-120h]
  __int64 v101; // [rsp+40h] [rbp-120h]
  _QWORD v102[2]; // [rsp+50h] [rbp-110h] BYREF
  __int64 v103[2]; // [rsp+60h] [rbp-100h] BYREF
  __int16 v104; // [rsp+70h] [rbp-F0h]
  __int64 v105; // [rsp+80h] [rbp-E0h] BYREF
  _BYTE *v106; // [rsp+88h] [rbp-D8h]
  _BYTE *v107; // [rsp+90h] [rbp-D0h]
  __int64 v108; // [rsp+98h] [rbp-C8h]
  int v109; // [rsp+A0h] [rbp-C0h]
  _BYTE v110[184]; // [rsp+A8h] [rbp-B8h] BYREF

  v6 = a3;
  v8 = &a3[a4];
  v9 = v110;
  v10 = (unsigned __int64)v110;
  v91 = a4;
  v105 = 0;
  v106 = v110;
  v107 = v110;
  v108 = 16;
  v109 = 0;
  if ( v8 != a3 )
  {
    do
    {
LABEL_5:
      v11 = *v6;
      if ( (_BYTE *)v10 != v9 )
        goto LABEL_3;
      v12 = (__int64 *)(v10 + 8LL * HIDWORD(v108));
      if ( v12 != (__int64 *)v10 )
      {
        v13 = (_QWORD *)v10;
        a4 = 0;
        while ( v11 != *v13 )
        {
          if ( *v13 == -2 )
            a4 = (__int64)v13;
          if ( v12 == ++v13 )
          {
            if ( !a4 )
              goto LABEL_159;
            ++v6;
            *(_QWORD *)a4 = v11;
            v10 = (unsigned __int64)v107;
            --v109;
            v9 = v106;
            ++v105;
            if ( v8 != v6 )
              goto LABEL_5;
            goto LABEL_14;
          }
        }
        goto LABEL_4;
      }
LABEL_159:
      if ( HIDWORD(v108) < (unsigned int)v108 )
      {
        ++HIDWORD(v108);
        *v12 = v11;
        v9 = v106;
        ++v105;
        v10 = (unsigned __int64)v107;
      }
      else
      {
LABEL_3:
        sub_16CCBA0((__int64)&v105, v11);
        v10 = (unsigned __int64)v107;
        v9 = v106;
      }
LABEL_4:
      ++v6;
    }
    while ( v8 != v6 );
  }
LABEL_14:
  for ( i = *(_QWORD *)(a1 + 48); ; i = v96 )
  {
    if ( !i )
      BUG();
    v15 = i - 24;
    if ( *(_BYTE *)(i - 8) != 77 )
      break;
    v96 = *(_QWORD *)(i + 8);
    if ( a6 )
      goto LABEL_56;
    v16 = *(_BYTE *)(i - 1);
    v17 = *a3;
    v18 = 0x17FFFFFFE8LL;
    v19 = v16 & 0x40;
    v20 = *(_DWORD *)(i - 4) & 0xFFFFFFF;
    if ( v20 )
    {
      v10 = v15 - 24LL * v20;
      v21 = 24LL * *(unsigned int *)(i + 32) + 8;
      v22 = 0;
      do
      {
        a4 = v15 - 24LL * v20;
        if ( v19 )
          a4 = *(_QWORD *)(i - 32);
        if ( v17 == *(_QWORD *)(a4 + v21) )
        {
          v18 = 24 * v22;
          goto LABEL_25;
        }
        ++v22;
        v21 += 8;
      }
      while ( v20 != (_DWORD)v22 );
      v18 = 0x17FFFFFFE8LL;
    }
LABEL_25:
    if ( v19 )
    {
      v23 = *(_QWORD *)(i - 32);
    }
    else
    {
      a4 = 24LL * v20;
      v23 = v15 - a4;
    }
    v24 = *(_QWORD *)(v23 + v18);
    if ( v20 )
    {
      v97 = i - 24;
      v25 = (unsigned __int64)v107;
      v26 = 0;
      v98 = 8LL * v20;
      for ( j = i; ; v16 = *(_BYTE *)(j - 1) )
      {
        if ( (v16 & 0x40) != 0 )
          v28 = *(_QWORD *)(j - 32);
        else
          v28 = v97 - 24LL * (*(_DWORD *)(j - 4) & 0xFFFFFFF);
        v29 = *(_QWORD *)(v26 + v28 + 24LL * *(unsigned int *)(j + 32) + 8);
        v30 = v106;
        if ( (_BYTE *)v25 == v106 )
        {
          v31 = (_QWORD *)(v25 + 8LL * HIDWORD(v108));
          if ( (_QWORD *)v25 == v31 )
          {
            v90 = (_QWORD *)v25;
          }
          else
          {
            do
            {
              if ( v29 == *v30 )
                break;
              ++v30;
            }
            while ( v31 != v30 );
            v90 = (_QWORD *)(v25 + 8LL * HIDWORD(v108));
          }
        }
        else
        {
          v31 = (_QWORD *)(v25 + 8LL * (unsigned int)v108);
          v30 = sub_16CC9F0((__int64)&v105, *(_QWORD *)(v26 + v28 + 24LL * *(unsigned int *)(j + 32) + 8));
          if ( v29 == *v30 )
          {
            v25 = (unsigned __int64)v107;
            if ( v107 == v106 )
              v90 = &v107[8 * HIDWORD(v108)];
            else
              v90 = &v107[8 * (unsigned int)v108];
          }
          else
          {
            v25 = (unsigned __int64)v107;
            if ( v107 != v106 )
            {
              v30 = &v107[8 * (unsigned int)v108];
              goto LABEL_34;
            }
            v30 = &v107[8 * HIDWORD(v108)];
            v90 = v30;
          }
        }
        while ( v90 != v30 && *v30 >= 0xFFFFFFFFFFFFFFFELL )
          ++v30;
LABEL_34:
        if ( v30 == v31 )
          goto LABEL_39;
        v32 = *(_BYTE *)(j - 1) & 0x40;
        if ( v24 )
        {
          if ( v32 )
          {
            v33 = *(_QWORD *)(*(_QWORD *)(j - 32) + 3 * v26);
            if ( !v33 )
              goto LABEL_55;
          }
          else
          {
            v33 = *(_QWORD *)(v97 - 24LL * (*(_DWORD *)(j - 4) & 0xFFFFFFF) + 3 * v26);
            if ( !v33 )
            {
LABEL_55:
              v15 = v97;
              i = j;
              goto LABEL_56;
            }
          }
          if ( v33 != v24 )
            goto LABEL_55;
LABEL_39:
          v26 += 8;
          if ( v26 == v98 )
            goto LABEL_85;
          continue;
        }
        if ( v32 )
          v60 = *(_QWORD *)(j - 32);
        else
          v60 = v97 - 24LL * (*(_DWORD *)(j - 4) & 0xFFFFFFF);
        v61 = 3 * v26;
        v26 += 8;
        v24 = *(_QWORD *)(v60 + v61);
        if ( v26 == v98 )
        {
LABEL_85:
          v15 = v97;
          i = j;
          break;
        }
      }
    }
    if ( v24 )
    {
      v100 = v24;
      v62 = i;
      v63 = (*(_DWORD *)(i - 4) & 0xFFFFFFFu) - 1;
      while ( 1 )
      {
        if ( (*(_BYTE *)(v62 - 1) & 0x40) != 0 )
          v64 = *(_QWORD *)(v62 - 32);
        else
          v64 = v15 - 24LL * (*(_DWORD *)(v62 - 4) & 0xFFFFFFF);
        v65 = *(_QWORD *)(v64 + 8 * v63 + 24LL * *(unsigned int *)(v62 + 32) + 8);
        v66 = v106;
        if ( v107 == v106 )
        {
          v67 = &v106[8 * HIDWORD(v108)];
          if ( v106 == v67 )
          {
            v68 = (__int64)v106;
          }
          else
          {
            do
            {
              if ( v65 == *v66 )
                break;
              ++v66;
            }
            while ( v67 != (_BYTE *)v66 );
            v68 = (__int64)&v106[8 * HIDWORD(v108)];
          }
          goto LABEL_137;
        }
        v64 = *(_QWORD *)(v64 + 8 * v63 + 24LL * *(unsigned int *)(v62 + 32) + 8);
        v67 = &v107[8 * (unsigned int)v108];
        v66 = sub_16CC9F0((__int64)&v105, v64);
        if ( v65 == *v66 )
          break;
        if ( v107 == v106 )
        {
          v66 = &v107[8 * HIDWORD(v108)];
          v68 = (__int64)v66;
          goto LABEL_137;
        }
        v68 = (unsigned int)v108;
        v66 = &v107[8 * (unsigned int)v108];
LABEL_93:
        if ( v67 != (_BYTE *)v66 )
        {
          v64 = (unsigned int)v63;
          sub_15F5350(v15, v63, 0);
        }
        v59 = v63-- == 0;
        if ( v59 )
        {
          v70 = v62;
          v69 = v100;
          v83 = *(_DWORD *)(v70 - 4) & 0xFFFFFFF;
          if ( v83 == *(_DWORD *)(v70 + 32) )
          {
            sub_15F55D0(v15, v64, v68, a4, v10, v17);
            v83 = *(_DWORD *)(v70 - 4) & 0xFFFFFFF;
          }
          v84 = (v83 + 1) & 0xFFFFFFF;
          v85 = v84 | *(_DWORD *)(v70 - 4) & 0xF0000000;
          *(_DWORD *)(v70 - 4) = v85;
          if ( (v85 & 0x40000000) != 0 )
            v86 = *(_QWORD *)(v70 - 32);
          else
            v86 = v15 - 24 * v84;
          v75 = (__int64 *)(v86 + 24LL * (unsigned int)(v84 - 1));
          if ( *v75 )
          {
            v87 = v75[1];
            v88 = v75[2] & 0xFFFFFFFFFFFFFFFCLL;
            *(_QWORD *)v88 = v87;
            if ( v87 )
              *(_QWORD *)(v87 + 16) = *(_QWORD *)(v87 + 16) & 3LL | v88;
          }
          *v75 = v100;
          v89 = *(_QWORD *)(v100 + 8);
          v80 = v100 + 8;
          v75[1] = v89;
          if ( v89 )
            *(_QWORD *)(v89 + 16) = (unsigned __int64)(v75 + 1) | *(_QWORD *)(v89 + 16) & 3LL;
          v79 = v75[2] & 3;
          goto LABEL_125;
        }
      }
      if ( v107 == v106 )
      {
        v64 = HIDWORD(v108);
        v68 = (__int64)&v107[8 * HIDWORD(v108)];
      }
      else
      {
        v64 = (unsigned int)v108;
        v68 = (__int64)&v107[8 * (unsigned int)v108];
      }
LABEL_137:
      while ( (_QWORD *)v68 != v66 && *v66 >= 0xFFFFFFFFFFFFFFFELL )
        ++v66;
      goto LABEL_93;
    }
LABEL_56:
    v102[0] = sub_1649960(v15);
    v104 = 773;
    v103[0] = (__int64)v102;
    v102[1] = v34;
    v103[1] = (__int64)".ph";
    v35 = *(_QWORD *)(i - 24);
    v36 = sub_1648B60(64);
    v38 = v36;
    if ( v36 )
    {
      sub_15F1EA0(v36, v35, 53, 0, 0, a5);
      *(_DWORD *)(v38 + 56) = v91;
      sub_164B780(v38, v103);
      sub_1648880(v38, *(_DWORD *)(v38 + 56), 1);
    }
    v39 = *(_DWORD *)(i - 4);
    v40 = i;
    v41 = v38;
    v42 = (v39 & 0xFFFFFFFu) - 1;
    v43 = v42;
    do
    {
      if ( (*(_BYTE *)(v40 - 1) & 0x40) != 0 )
        v44 = *(_QWORD *)(v40 - 32);
      else
        v44 = v15 - 24LL * (*(_DWORD *)(v40 - 4) & 0xFFFFFFF);
      v45 = *(_QWORD *)(v44 + 8 * v43 + 24LL * *(unsigned int *)(v40 + 32) + 8);
      v46 = v106;
      if ( v107 == v106 )
      {
        v47 = (__int64)&v106[8 * HIDWORD(v108)];
        if ( v106 == (_BYTE *)v47 )
        {
          v10 = (unsigned __int64)v106;
        }
        else
        {
          do
          {
            if ( v45 == *v46 )
              break;
            ++v46;
          }
          while ( (_QWORD *)v47 != v46 );
          v10 = (unsigned __int64)&v106[8 * HIDWORD(v108)];
        }
      }
      else
      {
        v44 = *(_QWORD *)(v44 + 8 * v43 + 24LL * *(unsigned int *)(v40 + 32) + 8);
        v99 = &v107[8 * (unsigned int)v108];
        v46 = sub_16CC9F0((__int64)&v105, v44);
        v10 = (unsigned __int64)v99;
        if ( v45 == *v46 )
        {
          if ( v107 == v106 )
          {
            v44 = HIDWORD(v108);
            v47 = (__int64)&v107[8 * HIDWORD(v108)];
          }
          else
          {
            v44 = (unsigned int)v108;
            v47 = (__int64)&v107[8 * (unsigned int)v108];
          }
        }
        else
        {
          if ( v107 != v106 )
          {
            v47 = (unsigned int)v108;
            v46 = &v107[8 * (unsigned int)v108];
            goto LABEL_64;
          }
          v47 = (__int64)&v107[8 * HIDWORD(v108)];
          v46 = (_QWORD *)v47;
        }
      }
      while ( (_QWORD *)v47 != v46 && *v46 >= 0xFFFFFFFFFFFFFFFELL )
        ++v46;
LABEL_64:
      if ( (_QWORD *)v10 != v46 )
      {
        v48 = sub_15F5350(v15, v43, 0);
        v49 = *(_DWORD *)(v41 + 20) & 0xFFFFFFF;
        if ( v49 == *(_DWORD *)(v41 + 56) )
        {
          v101 = v48;
          sub_15F55D0(v41, (unsigned int)v43, v48, v42, v10, v37);
          v48 = v101;
          v49 = *(_DWORD *)(v41 + 20) & 0xFFFFFFF;
        }
        v50 = (v49 + 1) & 0xFFFFFFF;
        v51 = v50 | *(_DWORD *)(v41 + 20) & 0xF0000000;
        *(_DWORD *)(v41 + 20) = v51;
        if ( (v51 & 0x40000000) != 0 )
          v52 = *(_QWORD *)(v41 - 8);
        else
          v52 = v41 - 24 * v50;
        v53 = (__int64 *)(v52 + 24LL * (unsigned int)(v50 - 1));
        if ( *v53 )
        {
          v54 = v53[1];
          v55 = v53[2] & 0xFFFFFFFFFFFFFFFCLL;
          *(_QWORD *)v55 = v54;
          if ( v54 )
          {
            v10 = *(_QWORD *)(v54 + 16) & 3LL;
            *(_QWORD *)(v54 + 16) = v10 | v55;
          }
        }
        *v53 = v48;
        if ( v48 )
        {
          v56 = *(_QWORD *)(v48 + 8);
          v10 = v48 + 8;
          v53[1] = v56;
          if ( v56 )
          {
            v37 = (__int64)(v53 + 1);
            *(_QWORD *)(v56 + 16) = (unsigned __int64)(v53 + 1) | *(_QWORD *)(v56 + 16) & 3LL;
          }
          v53[2] = v10 | v53[2] & 3;
          *(_QWORD *)(v48 + 8) = v53;
        }
        v57 = *(_DWORD *)(v41 + 20) & 0xFFFFFFF;
        v58 = (unsigned int)(v57 - 1);
        if ( (*(_BYTE *)(v41 + 23) & 0x40) != 0 )
          v44 = *(_QWORD *)(v41 - 8);
        else
          v44 = v41 - 24 * v57;
        v47 = 3LL * *(unsigned int *)(v41 + 56);
        *(_QWORD *)(v44 + 8 * v58 + 24LL * *(unsigned int *)(v41 + 56) + 8) = v45;
      }
      v59 = v43-- == 0;
    }
    while ( !v59 );
    v69 = v41;
    v70 = v40;
    v71 = *(_DWORD *)(v40 - 4) & 0xFFFFFFF;
    if ( v71 == *(_DWORD *)(v40 + 32) )
    {
      sub_15F55D0(v15, v44, v47, v42, v10, v37);
      v71 = *(_DWORD *)(v40 - 4) & 0xFFFFFFF;
    }
    v72 = (v71 + 1) & 0xFFFFFFF;
    v73 = v72 | *(_DWORD *)(v40 - 4) & 0xF0000000;
    *(_DWORD *)(v40 - 4) = v73;
    if ( (v73 & 0x40000000) != 0 )
      v74 = *(_QWORD *)(v40 - 32);
    else
      v74 = v15 - 24 * v72;
    v75 = (__int64 *)(v74 + 24LL * (unsigned int)(v72 - 1));
    if ( *v75 )
    {
      v76 = v75[1];
      v77 = v75[2] & 0xFFFFFFFFFFFFFFFCLL;
      *(_QWORD *)v77 = v76;
      if ( v76 )
        *(_QWORD *)(v76 + 16) = *(_QWORD *)(v76 + 16) & 3LL | v77;
    }
    *v75 = v69;
    if ( v69 )
    {
      v78 = *(_QWORD *)(v69 + 8);
      v79 = v69 + 8;
      v75[1] = v78;
      if ( v78 )
        *(_QWORD *)(v78 + 16) = (unsigned __int64)(v75 + 1) | *(_QWORD *)(v78 + 16) & 3LL;
      v80 = v75[2] & 3;
LABEL_125:
      v75[2] = v79 | v80;
      *(_QWORD *)(v69 + 8) = v75;
    }
    v81 = *(_DWORD *)(v70 - 4) & 0xFFFFFFF;
    if ( (*(_BYTE *)(v70 - 1) & 0x40) != 0 )
      v82 = *(_QWORD *)(v70 - 32);
    else
      v82 = v15 - 24 * v81;
    a4 = a2;
    *(_QWORD *)(v82 + 8LL * (unsigned int)(v81 - 1) + 24LL * *(unsigned int *)(v70 + 32) + 8) = a2;
  }
  if ( v107 != v106 )
    _libc_free((unsigned __int64)v107);
}
