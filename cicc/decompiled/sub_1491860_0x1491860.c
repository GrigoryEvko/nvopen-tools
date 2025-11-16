// Function: sub_1491860
// Address: 0x1491860
//
__int64 __fastcall sub_1491860(__int64 a1, __int64 a2, __int64 a3, __m128i a4, __m128i a5)
{
  __int64 v5; // r14
  __int64 v6; // r13
  __int64 v7; // rax
  int v8; // ecx
  __int64 v9; // rsi
  __int64 v10; // rdi
  int v11; // ecx
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r10
  _QWORD *v15; // rdi
  __int64 v16; // r15
  __int64 v17; // rbx
  __int64 v18; // r13
  __int64 v19; // rdx
  __int64 v20; // r12
  __int64 v21; // rdx
  _QWORD *v22; // rax
  _QWORD *v23; // r14
  _BYTE *v24; // rdi
  __int64 v26; // rsi
  __int64 v27; // r14
  __int64 v28; // rax
  __int64 v29; // r10
  __int64 v30; // rax
  int v31; // r15d
  __int64 v32; // r12
  __int64 v33; // r15
  __int64 v34; // rbx
  __int64 v35; // rax
  __int16 v36; // ax
  __int64 v37; // rax
  bool v38; // zf
  __int64 v39; // rbx
  __int64 v40; // r13
  __int64 v41; // r12
  __int64 *v42; // rdx
  __int64 v43; // rbx
  __int64 *v44; // r12
  __int64 v45; // rdx
  unsigned int v46; // r14d
  __int64 v47; // r13
  __int64 v48; // r15
  __int64 v49; // rax
  __int64 v50; // rsi
  __int64 v51; // r15
  __int64 v52; // rax
  __int64 v53; // r15
  __int64 v54; // rbx
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // rbx
  _QWORD *v58; // r15
  __int64 v59; // rax
  __int64 v60; // rax
  __int64 v61; // r13
  char v62; // r9
  __int64 *v63; // rax
  int v64; // esi
  int v65; // edx
  unsigned int v66; // esi
  __int64 v67; // rdx
  int v68; // eax
  int v69; // eax
  __int64 v70; // r15
  __int64 v71; // rax
  __int64 v72; // rax
  int v73; // r8d
  _QWORD *v74; // rbx
  __int64 v75; // rax
  bool v76; // [rsp+7h] [rbp-149h]
  __int64 v80; // [rsp+28h] [rbp-128h]
  __int64 v81; // [rsp+28h] [rbp-128h]
  __int64 v82; // [rsp+30h] [rbp-120h]
  __int64 v83; // [rsp+30h] [rbp-120h]
  __int64 v84; // [rsp+30h] [rbp-120h]
  __int64 v85; // [rsp+38h] [rbp-118h]
  __int64 v86; // [rsp+38h] [rbp-118h]
  __int64 v87; // [rsp+40h] [rbp-110h]
  __int64 v88; // [rsp+48h] [rbp-108h]
  int v89; // [rsp+48h] [rbp-108h]
  __int64 *v90; // [rsp+48h] [rbp-108h]
  __int64 *v91; // [rsp+58h] [rbp-F8h] BYREF
  __int64 v92; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v93; // [rsp+68h] [rbp-E8h]
  _BYTE *v94; // [rsp+70h] [rbp-E0h] BYREF
  __int64 v95; // [rsp+78h] [rbp-D8h]
  _BYTE v96[32]; // [rsp+80h] [rbp-D0h] BYREF
  __int64 v97; // [rsp+A0h] [rbp-B0h] BYREF
  char *v98; // [rsp+A8h] [rbp-A8h] BYREF
  __int64 v99; // [rsp+B0h] [rbp-A0h]
  _BYTE v100[24]; // [rsp+B8h] [rbp-98h] BYREF
  __int64 v101[2]; // [rsp+D0h] [rbp-80h] BYREF
  _BYTE v102[112]; // [rsp+E0h] [rbp-70h] BYREF

  v5 = 0;
  v6 = *(_QWORD *)(a3 + 24);
  v94 = v96;
  v95 = 0x300000000LL;
  if ( *(_BYTE *)(*(_QWORD *)v6 + 8LL) == 11 )
  {
    v7 = *(_QWORD *)(a2 + 64);
    v8 = *(_DWORD *)(v7 + 24);
    if ( v8 )
    {
      v9 = *(_QWORD *)(v6 + 40);
      v10 = *(_QWORD *)(v7 + 8);
      v11 = v8 - 1;
      v12 = v11 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v13 = (__int64 *)(v10 + 16LL * v12);
      v14 = *v13;
      if ( v9 == *v13 )
      {
LABEL_4:
        v5 = v13[1];
        if ( v5 && v9 != **(_QWORD **)(v5 + 32) )
          v5 = 0;
      }
      else
      {
        v69 = 1;
        while ( v14 != -8 )
        {
          v73 = v69 + 1;
          v12 = v11 & (v69 + v12);
          v13 = (__int64 *)(v10 + 16LL * v12);
          v14 = *v13;
          if ( v9 == *v13 )
            goto LABEL_4;
          v69 = v73;
        }
        v5 = 0;
      }
    }
  }
  if ( (*(_DWORD *)(v6 + 20) & 0xFFFFFFF) == 0 )
  {
    v24 = v96;
    goto LABEL_23;
  }
  v15 = *(_QWORD **)(v5 + 72);
  v16 = v5;
  v87 = 0;
  v82 = 0;
  v85 = 8LL * (*(_DWORD *)(v6 + 20) & 0xFFFFFFF);
  v80 = v5 + 56;
  v17 = v6;
  v18 = 0;
  do
  {
    while ( 1 )
    {
      v19 = (*(_BYTE *)(v17 + 23) & 0x40) != 0 ? *(_QWORD *)(v17 - 8) : v17 - 24LL * (*(_DWORD *)(v17 + 20) & 0xFFFFFFF);
      v20 = *(_QWORD *)(v19 + 3 * v18);
      v21 = *(_QWORD *)(v18 + v19 + 24LL * *(unsigned int *)(v17 + 56) + 8);
      v22 = *(_QWORD **)(v16 + 64);
      if ( v15 == v22 )
      {
        v26 = *(unsigned int *)(v16 + 84);
        v23 = &v15[v26];
        if ( v15 == v23 )
        {
          v45 = (__int64)v15;
        }
        else
        {
          do
          {
            if ( v21 == *v22 )
              break;
            ++v22;
          }
          while ( v23 != v22 );
          v45 = (__int64)&v15[v26];
        }
      }
      else
      {
        v88 = v21;
        v23 = &v15[*(unsigned int *)(v16 + 80)];
        v22 = (_QWORD *)sub_16CC9F0(v80, v21);
        if ( v88 == *v22 )
        {
          v15 = *(_QWORD **)(v16 + 72);
          v45 = (__int64)(v15 == *(_QWORD **)(v16 + 64)
                        ? &v15[*(unsigned int *)(v16 + 84)]
                        : &v15[*(unsigned int *)(v16 + 80)]);
        }
        else
        {
          v15 = *(_QWORD **)(v16 + 72);
          if ( v15 != *(_QWORD **)(v16 + 64) )
          {
            v22 = &v15[*(unsigned int *)(v16 + 80)];
            goto LABEL_14;
          }
          v22 = &v15[*(unsigned int *)(v16 + 84)];
          v45 = (__int64)v22;
        }
      }
      while ( (_QWORD *)v45 != v22 && *v22 >= 0xFFFFFFFFFFFFFFFELL )
        ++v22;
LABEL_14:
      if ( v23 != v22 )
        break;
      if ( v87 )
      {
        if ( v87 != v20 )
          goto LABEL_22;
      }
      else
      {
        v87 = v20;
      }
LABEL_17:
      v18 += 8;
      if ( v85 == v18 )
        goto LABEL_39;
    }
    if ( v82 )
    {
      if ( v20 != v82 )
        goto LABEL_22;
      goto LABEL_17;
    }
    v82 = v20;
    v18 += 8;
  }
  while ( v85 != v18 );
LABEL_39:
  v27 = v16;
  if ( !v82 || !v87 )
  {
LABEL_22:
    v24 = v94;
LABEL_23:
    *(_BYTE *)(a1 + 48) = 0;
    goto LABEL_24;
  }
  v28 = sub_146F1B0(a2, v82);
  v29 = v28;
  if ( *(_WORD *)(v28 + 24) != 4 )
    goto LABEL_65;
  v30 = *(_QWORD *)(v28 + 40);
  v31 = v30;
  if ( (_DWORD)v30 )
  {
    v32 = 0;
    v33 = v29;
    v83 = (unsigned int)v30;
    v81 = v27;
    v86 = a3 + 32;
    while ( 1 )
    {
      v34 = *(_QWORD *)(*(_QWORD *)(v33 + 32) + 8 * v32);
      if ( v34 != v86 )
      {
        v89 = sub_1456C90(a2, **(_QWORD **)(a3 + 24));
        v35 = sub_1456040(v34);
        if ( v89 == (unsigned int)sub_1456C90(a2, v35) )
        {
          v36 = *(_WORD *)(v34 + 24);
          if ( v36 == 3 )
          {
            v37 = *(_QWORD *)(v34 + 32);
            if ( *(_WORD *)(v37 + 24) == 1 )
              goto LABEL_48;
          }
          else if ( v36 == 2 )
          {
            v37 = *(_QWORD *)(v34 + 32);
            v34 = 0;
            if ( *(_WORD *)(v37 + 24) == 1 )
            {
LABEL_48:
              if ( *(_QWORD *)(v37 + 32) == v86 )
              {
                v38 = v34 == 0;
                v39 = *(_QWORD *)(v37 + 40);
                v76 = !v38;
                if ( v39 )
                {
                  v29 = v33;
                  v31 = v32;
                  v27 = v81;
                  v30 = *(_QWORD *)(v29 + 40);
                  goto LABEL_51;
                }
              }
            }
          }
        }
      }
      v46 = ++v32;
      if ( v83 == v32 )
      {
        v29 = v33;
        v32 = v46;
        v39 = 0;
        v27 = v81;
        v30 = *(_QWORD *)(v33 + 40);
        v31 = v32;
        goto LABEL_51;
      }
    }
  }
  v32 = 0;
  v39 = 0;
LABEL_51:
  if ( v30 == v32 )
  {
LABEL_65:
    *(_BYTE *)(a1 + 48) = 0;
    v24 = v94;
    goto LABEL_24;
  }
  v40 = (unsigned int)v30;
  v41 = 0;
  v42 = &v97;
  v101[0] = (__int64)v102;
  v101[1] = 0x800000000LL;
  if ( (_DWORD)v30 )
  {
    v84 = v39;
    v43 = v29;
    do
    {
      if ( v31 != (_DWORD)v41 )
      {
        v90 = v42;
        v97 = *(_QWORD *)(*(_QWORD *)(v43 + 32) + 8 * v41);
        sub_1458920((__int64)v101, v42);
        v42 = v90;
      }
      ++v41;
    }
    while ( v41 != v40 );
    v39 = v84;
  }
  v44 = sub_147DD40(a2, v101, 0, 0, a4, a5);
  if ( !sub_146CEE0(a2, (__int64)v44, v27) )
  {
LABEL_59:
    *(_BYTE *)(a1 + 48) = 0;
    goto LABEL_60;
  }
  v47 = sub_146F1B0(a2, v87);
  v48 = sub_14835F0((_QWORD *)a2, (__int64)v44, v39, 0, a4, a5);
  v49 = sub_14835F0((_QWORD *)a2, v47, v39, 0, a4, a5);
  v50 = sub_14799E0(a2, v49, v48, v27, 0);
  if ( *(_WORD *)(v50 + 24) == 7 )
  {
    *(_QWORD *)&v94[8 * (unsigned int)v95] = sub_145DF90(a2, v50, 1 - ((unsigned int)!v76 - 1));
    LODWORD(v95) = v95 + 1;
  }
  v51 = sub_14835F0((_QWORD *)a2, v47, v39, 0, a4, a5);
  v52 = sub_1456040(v47);
  if ( v76 )
    v53 = sub_147B0D0(a2, v51, v52, 0);
  else
    v53 = sub_14747F0(a2, v51, v52, 0);
  if ( v47 == v53 )
  {
    v70 = sub_14835F0((_QWORD *)a2, (__int64)v44, v39, 0, a4, a5);
    v71 = sub_1456040((__int64)v44);
    v72 = sub_147B0D0(a2, v70, v71, 0);
    v57 = v72;
    if ( v44 == (__int64 *)v72 )
      goto LABEL_82;
    if ( (unsigned __int8)sub_147A340(a2, 0x21u, (__int64)v44, v72) )
      goto LABEL_59;
    goto LABEL_81;
  }
  if ( (unsigned __int8)sub_147A340(a2, 0x21u, v47, v53) )
    goto LABEL_59;
  v54 = sub_14835F0((_QWORD *)a2, (__int64)v44, v39, 0, a4, a5);
  v55 = sub_1456040((__int64)v44);
  v56 = sub_147B0D0(a2, v54, v55, 0);
  v57 = v56;
  if ( v44 != (__int64 *)v56 )
  {
    if ( (unsigned __int8)sub_147A340(a2, 0x21u, (__int64)v44, v56) )
      goto LABEL_59;
    if ( !(unsigned __int8)sub_147A340(a2, 0x20u, v47, v53) )
    {
LABEL_78:
      v58 = sub_145DE40(a2, v47, v53);
      v59 = (unsigned int)v95;
      if ( (unsigned int)v95 >= HIDWORD(v95) )
      {
        sub_16CD150(&v94, v96, 0, 8);
        v59 = (unsigned int)v95;
      }
      *(_QWORD *)&v94[8 * v59] = v58;
      LODWORD(v95) = v95 + 1;
      if ( v44 == (__int64 *)v57 )
        goto LABEL_82;
    }
LABEL_81:
    if ( !(unsigned __int8)sub_147A340(a2, 0x20u, (__int64)v44, v57) )
    {
      v74 = sub_145DE40(a2, (__int64)v44, v57);
      v75 = (unsigned int)v95;
      if ( (unsigned int)v95 >= HIDWORD(v95) )
      {
        sub_16CD150(&v94, v96, 0, 8);
        v75 = (unsigned int)v95;
      }
      *(_QWORD *)&v94[8 * v75] = v74;
      LODWORD(v95) = v95 + 1;
    }
    goto LABEL_82;
  }
  if ( !(unsigned __int8)sub_147A340(a2, 0x20u, v47, v53) )
    goto LABEL_78;
LABEL_82:
  v60 = sub_14799E0(a2, v47, (__int64)v44, v27, 0);
  v98 = v100;
  v97 = v60;
  v99 = 0x300000000LL;
  if ( (_DWORD)v95 )
    sub_14531E0((__int64)&v98, (__int64)&v94);
  v93 = v27;
  v61 = a2 + 1000;
  v92 = a3;
  v62 = sub_145F9E0(a2 + 1000, &v92, &v91);
  v63 = v91;
  if ( !v62 )
  {
    v64 = *(_DWORD *)(a2 + 1016);
    ++*(_QWORD *)(a2 + 1000);
    v65 = v64 + 1;
    v66 = *(_DWORD *)(a2 + 1024);
    if ( 4 * v65 >= 3 * v66 )
    {
      v66 *= 2;
    }
    else if ( v66 - *(_DWORD *)(a2 + 1020) - v65 > v66 >> 3 )
    {
LABEL_87:
      *(_DWORD *)(a2 + 1016) = v65;
      if ( *v63 != -8 || v63[1] != -8 )
        --*(_DWORD *)(a2 + 1020);
      v67 = v92;
      v63[2] = 0;
      v63[4] = 0x300000000LL;
      *v63 = v67;
      v63[1] = v93;
      v63[3] = (__int64)(v63 + 5);
      goto LABEL_90;
    }
    sub_1468310(v61, v66);
    sub_145F9E0(v61, &v92, &v91);
    v63 = v91;
    v65 = *(_DWORD *)(a2 + 1016) + 1;
    goto LABEL_87;
  }
LABEL_90:
  v63[2] = v97;
  sub_14531E0((__int64)(v63 + 3), (__int64)&v98);
  *(_QWORD *)a1 = v97;
  *(_QWORD *)(a1 + 8) = a1 + 24;
  *(_QWORD *)(a1 + 16) = 0x300000000LL;
  v68 = v99;
  *(_BYTE *)(a1 + 48) = 1;
  if ( v68 )
    sub_14532C0(a1 + 8, &v98);
  if ( v98 != v100 )
    _libc_free((unsigned __int64)v98);
LABEL_60:
  if ( (_BYTE *)v101[0] != v102 )
    _libc_free(v101[0]);
  v24 = v94;
LABEL_24:
  if ( v24 != v96 )
    _libc_free((unsigned __int64)v24);
  return a1;
}
