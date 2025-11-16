// Function: sub_2728E40
// Address: 0x2728e40
//
__int64 __fastcall sub_2728E40(__int64 a1, __int64 a2)
{
  __int64 v2; // rdx
  __int64 v3; // r14
  __int64 v4; // r12
  __int64 v5; // r13
  __int64 i; // r14
  unsigned __int8 *v7; // rbx
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned __int8 **v16; // r12
  unsigned __int8 **v17; // rbx
  __int64 v18; // rcx
  __int64 v19; // rax
  __int64 v20; // rdx
  unsigned __int8 **v21; // rbx
  _QWORD *v22; // rdi
  __int64 v24; // rcx
  int v25; // edx
  int v26; // eax
  unsigned int v27; // ecx
  unsigned int v28; // eax
  __int64 v29; // rdx
  unsigned __int64 v30; // rax
  __int64 v31; // r8
  __int64 v32; // r9
  __int64 v33; // rdx
  int v34; // eax
  __int64 v35; // rcx
  __int64 v36; // rdi
  __int64 *v37; // rsi
  unsigned __int8 v38; // al
  __int64 v39; // rax
  unsigned __int8 *v40; // r15
  unsigned __int8 *v41; // rcx
  __int64 *v42; // r12
  __int64 v43; // r15
  __int64 *v44; // rbx
  __int64 v45; // rsi
  int v46; // edx
  unsigned __int8 v47; // al
  __int64 v48; // rdx
  __int64 v49; // rcx
  __int64 v50; // r8
  __int64 v51; // r9
  unsigned __int8 v52; // r14
  __int64 v53; // rax
  _BYTE *v54; // rdx
  __int64 v55; // rdx
  unsigned int v56; // eax
  __int64 v57; // rdx
  unsigned __int64 v58; // r10
  __int64 (__fastcall *v59)(__int64, unsigned int, _BYTE *, __int64); // rax
  __int64 v60; // rax
  __int64 v61; // r15
  __int64 v62; // r8
  __int64 v63; // r9
  __int64 v64; // rax
  unsigned __int64 v65; // rdx
  unsigned __int64 v66; // rdi
  _BYTE *v67; // rax
  __int64 v68; // r8
  __int64 v69; // r9
  __int64 v70; // rax
  unsigned __int64 v71; // rdx
  unsigned int v72; // edx
  int v73; // eax
  char v74; // r15
  __int64 v75; // rax
  _QWORD *v76; // rax
  unsigned __int64 v77; // r14
  unsigned __int64 v78; // rbx
  __int64 v79; // rdx
  unsigned int v80; // esi
  unsigned __int64 v81; // [rsp+0h] [rbp-580h]
  unsigned __int64 v82; // [rsp+0h] [rbp-580h]
  __int64 v83; // [rsp+0h] [rbp-580h]
  unsigned __int8 *v84; // [rsp+0h] [rbp-580h]
  unsigned int v85; // [rsp+18h] [rbp-568h]
  __int64 **v86; // [rsp+20h] [rbp-560h]
  __int64 v87; // [rsp+20h] [rbp-560h]
  __int64 v88; // [rsp+20h] [rbp-560h]
  int v89; // [rsp+28h] [rbp-558h]
  unsigned int v90; // [rsp+28h] [rbp-558h]
  __int64 v91; // [rsp+28h] [rbp-558h]
  unsigned int v92; // [rsp+28h] [rbp-558h]
  unsigned __int8 v93; // [rsp+30h] [rbp-550h]
  unsigned __int64 v95; // [rsp+40h] [rbp-540h] BYREF
  unsigned int v96; // [rsp+48h] [rbp-538h]
  _QWORD v97[4]; // [rsp+50h] [rbp-530h] BYREF
  __int16 v98; // [rsp+70h] [rbp-510h]
  char v99[32]; // [rsp+80h] [rbp-500h] BYREF
  __int16 v100; // [rsp+A0h] [rbp-4E0h]
  unsigned __int64 v101; // [rsp+B0h] [rbp-4D0h] BYREF
  __int64 v102; // [rsp+B8h] [rbp-4C8h]
  _BYTE v103[32]; // [rsp+C0h] [rbp-4C0h] BYREF
  __int64 v104; // [rsp+E0h] [rbp-4A0h]
  __int64 v105; // [rsp+E8h] [rbp-498h]
  __int64 v106; // [rsp+F0h] [rbp-490h]
  __int64 v107; // [rsp+F8h] [rbp-488h]
  void **v108; // [rsp+100h] [rbp-480h]
  void **v109; // [rsp+108h] [rbp-478h]
  __int64 v110; // [rsp+110h] [rbp-470h]
  int v111; // [rsp+118h] [rbp-468h]
  __int16 v112; // [rsp+11Ch] [rbp-464h]
  char v113; // [rsp+11Eh] [rbp-462h]
  __int64 v114; // [rsp+120h] [rbp-460h]
  __int64 v115; // [rsp+128h] [rbp-458h]
  void *v116; // [rsp+130h] [rbp-450h] BYREF
  void *v117; // [rsp+138h] [rbp-448h] BYREF
  unsigned __int8 **v118; // [rsp+140h] [rbp-440h] BYREF
  __int64 v119; // [rsp+148h] [rbp-438h]
  _BYTE v120[1072]; // [rsp+150h] [rbp-430h] BYREF

  v2 = a1 + 72;
  v3 = *(_QWORD *)(a1 + 80);
  v118 = (unsigned __int8 **)v120;
  v119 = 0x8000000000LL;
  if ( a1 + 72 == v3 )
    return 0;
  if ( !v3 )
    BUG();
  while ( *(_QWORD *)(v3 + 32) == v3 + 24 )
  {
    v3 = *(_QWORD *)(v3 + 8);
    if ( v2 == v3 )
      return 0;
    if ( !v3 )
      BUG();
  }
  v4 = a1 + 72;
  if ( v3 == v2 )
    return 0;
  v93 = 0;
  v5 = v3;
  i = *(_QWORD *)(v3 + 32);
  do
  {
    v7 = (unsigned __int8 *)(i - 24);
    if ( !i )
      v7 = 0;
    if ( (unsigned __int8)sub_B46970(v7) && !*((_QWORD *)v7 + 2) )
      goto LABEL_18;
    if ( (unsigned __int8)sub_D198A0(a2, v7, v8, v9, v10, v11) )
    {
LABEL_15:
      v14 = (unsigned int)v119;
      v15 = (unsigned int)v119 + 1LL;
      if ( v15 > HIDWORD(v119) )
      {
        sub_C8D5F0((__int64)&v118, v120, v15, 8u, v12, v13);
        v14 = (unsigned int)v119;
      }
      v93 = 1;
      v118[v14] = v7;
      LODWORD(v119) = v119 + 1;
      goto LABEL_18;
    }
    v24 = *((_QWORD *)v7 + 1);
    v25 = *(unsigned __int8 *)(v24 + 8);
    if ( (unsigned int)(v25 - 17) <= 1 )
      LOBYTE(v25) = *(_BYTE *)(**(_QWORD **)(v24 + 16) + 8LL);
    if ( (_BYTE)v25 == 12 )
    {
      sub_D19730((__int64)&v101, a2, (__int64)v7, v24, v12, v13);
      v72 = v102;
      if ( (unsigned int)v102 <= 0x40 )
      {
        v74 = v101 == 0;
      }
      else
      {
        v92 = v102;
        v73 = sub_C444A0((__int64)&v101);
        v72 = v92;
        v74 = v92 == v73;
      }
      if ( v74 )
      {
        v74 = sub_F509B0(v7, 0);
        if ( (unsigned int)v102 <= 0x40 )
          goto LABEL_119;
      }
      else if ( v72 <= 0x40 )
      {
        goto LABEL_43;
      }
      if ( v101 )
        j_j___libc_free_0_0(v101);
LABEL_119:
      if ( v74 )
        goto LABEL_15;
    }
LABEL_43:
    v26 = *v7;
    if ( (_BYTE)v26 != 69 )
      goto LABEL_49;
    sub_D19730((__int64)&v95, a2, (__int64)v7, v24, v12, v13);
    v89 = sub_BCB060(*(_QWORD *)(*((_QWORD *)v7 - 4) + 8LL));
    v86 = (__int64 **)*((_QWORD *)v7 + 1);
    v27 = sub_BCB060((__int64)v86);
    v28 = v96;
    if ( v96 > 0x40 )
    {
      v85 = v27;
      v56 = sub_C444A0((__int64)&v95);
      v24 = v85;
      v12 = v56;
      if ( v56 >= v85 - v89 )
        goto LABEL_86;
      if ( v95 )
        j_j___libc_free_0_0(v95);
LABEL_48:
      v26 = *v7;
LABEL_49:
      if ( (unsigned int)(v26 - 42) > 0x11 )
        goto LABEL_60;
      sub_D19730((__int64)&v101, a2, (__int64)v7, v24, v12, v13);
      v33 = (unsigned int)v102;
      if ( !(_DWORD)v102 )
        goto LABEL_60;
      if ( (unsigned int)v102 <= 0x40 )
      {
        v35 = (unsigned int)(64 - v102);
        if ( v101 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v102) )
          goto LABEL_60;
      }
      else
      {
        v90 = v102;
        v34 = sub_C445E0((__int64)&v101);
        v33 = v90;
        if ( v90 == v34 )
          goto LABEL_58;
      }
      v36 = *((_QWORD *)v7 - 4);
      v37 = (__int64 *)(v36 + 24);
      if ( *(_BYTE *)v36 == 17 )
      {
        v38 = *v7;
        if ( *v7 != 57 )
          goto LABEL_55;
        goto LABEL_106;
      }
      if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v36 + 8) + 8LL) - 17 > 1 || *(_BYTE *)v36 > 0x15u )
      {
LABEL_84:
        if ( (unsigned int)v33 <= 0x40 )
          goto LABEL_60;
      }
      else
      {
        v67 = sub_AD7630(v36, 0, v33);
        if ( v67 && *v67 == 17 )
        {
          v37 = (__int64 *)(v67 + 24);
          v38 = *v7;
          v33 = (unsigned int)v102;
          if ( *v7 != 57 )
          {
LABEL_55:
            if ( (unsigned __int8)(v38 - 58) > 1u )
              goto LABEL_84;
            if ( (unsigned int)v33 <= 0x40 )
            {
              if ( (*v37 & v101) != 0 )
                goto LABEL_60;
            }
            else if ( (unsigned __int8)sub_C446A0((__int64 *)&v101, v37) )
            {
              goto LABEL_58;
            }
            goto LABEL_108;
          }
LABEL_106:
          if ( (unsigned int)v33 <= 0x40 )
          {
            if ( (v101 & ~*v37) != 0 )
              goto LABEL_60;
          }
          else if ( !(unsigned __int8)sub_C446F0((__int64 *)&v101, v37) )
          {
            goto LABEL_58;
          }
LABEL_108:
          sub_2728DA0((__int64)v7, a2, v33, v35, v31, v32);
          sub_BD84D0((__int64)v7, *((_QWORD *)v7 - 8));
          v70 = (unsigned int)v119;
          v71 = (unsigned int)v119 + 1LL;
          if ( v71 > HIDWORD(v119) )
          {
            sub_C8D5F0((__int64)&v118, v120, v71, 8u, v68, v69);
            v70 = (unsigned int)v119;
          }
          v118[v70] = v7;
          LODWORD(v119) = v119 + 1;
          if ( (unsigned int)v102 > 0x40 )
          {
            v66 = v101;
            if ( v101 )
              goto LABEL_112;
          }
          goto LABEL_99;
        }
        if ( (unsigned int)v102 <= 0x40 )
        {
LABEL_60:
          v39 = 32LL * (*((_DWORD *)v7 + 1) & 0x7FFFFFF);
          if ( (v7[7] & 0x40) != 0 )
          {
            v40 = (unsigned __int8 *)*((_QWORD *)v7 - 1);
            v41 = &v40[v39];
          }
          else
          {
            v41 = v7;
            v40 = &v7[-v39];
          }
          if ( v40 != v41 )
          {
            v91 = i;
            v87 = v4;
            v42 = (__int64 *)v40;
            v43 = (__int64)v7;
            v44 = (__int64 *)v41;
            do
            {
              v45 = *(_QWORD *)(*v42 + 8);
              v46 = *(unsigned __int8 *)(v45 + 8);
              if ( (unsigned int)(v46 - 17) <= 1 )
                LOBYTE(v46) = *(_BYTE *)(**(_QWORD **)(v45 + 16) + 8LL);
              if ( (_BYTE)v46 == 12 )
              {
                v47 = *(_BYTE *)*v42;
                if ( v47 == 22 || v47 > 0x1Cu )
                {
                  v52 = sub_D19970(a2, v42);
                  if ( v52 )
                  {
                    sub_2728DA0(v43, a2, v48, v49, v50, v51);
                    v53 = sub_AD64C0(*(_QWORD *)(*v42 + 8), 0, 0);
                    if ( *v42 )
                    {
                      v54 = (_BYTE *)v42[1];
                      *(_QWORD *)v42[2] = v54;
                      if ( v54 )
                        *((_QWORD *)v54 + 2) = v42[2];
                    }
                    *v42 = v53;
                    v93 = v52;
                    if ( v53 )
                    {
                      v55 = *(_QWORD *)(v53 + 16);
                      v42[1] = v55;
                      if ( v55 )
                        *(_QWORD *)(v55 + 16) = v42 + 1;
                      v42[2] = v53 + 16;
                      v93 = v52;
                      *(_QWORD *)(v53 + 16) = v42;
                    }
                  }
                }
              }
              v42 += 4;
            }
            while ( v44 != v42 );
            i = v91;
            v4 = v87;
          }
          goto LABEL_18;
        }
      }
LABEL_58:
      if ( v101 )
        j_j___libc_free_0_0(v101);
      goto LABEL_60;
    }
    v29 = v95;
    if ( v95 )
    {
      _BitScanReverse64(&v30, v95);
      v28 = v96 - 64 + (v30 ^ 0x3F);
    }
    v24 = v27 - v89;
    if ( v28 < (unsigned int)v24 )
      goto LABEL_48;
LABEL_86:
    sub_2728DA0((__int64)v7, a2, v29, v24, v12, v13);
    v107 = sub_BD5C60((__int64)v7);
    v108 = &v116;
    v109 = &v117;
    v101 = (unsigned __int64)v103;
    v116 = &unk_49DA100;
    v102 = 0x200000000LL;
    v112 = 512;
    LOWORD(v106) = 0;
    v117 = &unk_49DA0B0;
    v110 = 0;
    v111 = 0;
    v113 = 7;
    v114 = 0;
    v115 = 0;
    v104 = 0;
    v105 = 0;
    sub_D5F1F0((__int64)&v101, (__int64)v7);
    v97[0] = sub_BD5D20((__int64)v7);
    v98 = 261;
    v97[1] = v57;
    v58 = *((_QWORD *)v7 - 4);
    if ( v86 == *(__int64 ***)(v58 + 8) )
    {
      v61 = *((_QWORD *)v7 - 4);
      goto LABEL_93;
    }
    v59 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, __int64))*((_QWORD *)*v108 + 15);
    if ( v59 != sub_920130 )
    {
      v82 = *((_QWORD *)v7 - 4);
      v75 = v59((__int64)v108, 39u, (_BYTE *)v58, (__int64)v86);
      v58 = v82;
      v61 = v75;
      goto LABEL_92;
    }
    if ( *(_BYTE *)v58 <= 0x15u )
    {
      v81 = *((_QWORD *)v7 - 4);
      if ( (unsigned __int8)sub_AC4810(0x27u) )
        v60 = sub_ADAB70(39, v81, v86, 0);
      else
        v60 = sub_AA93C0(0x27u, v81, (__int64)v86);
      v58 = v81;
      v61 = v60;
LABEL_92:
      if ( v61 )
        goto LABEL_93;
    }
    v83 = v58;
    v100 = 257;
    v76 = sub_BD2C40(72, unk_3F10A14);
    v61 = (__int64)v76;
    if ( v76 )
      sub_B515B0((__int64)v76, v83, (__int64)v86, (__int64)v99, 0, 0);
    (*((void (__fastcall **)(void **, __int64, _QWORD *, __int64, __int64))*v109 + 2))(v109, v61, v97, v105, v106);
    if ( v101 != v101 + 16LL * (unsigned int)v102 )
    {
      v88 = i;
      v77 = v101 + 16LL * (unsigned int)v102;
      v84 = v7;
      v78 = v101;
      do
      {
        v79 = *(_QWORD *)(v78 + 8);
        v80 = *(_DWORD *)v78;
        v78 += 16LL;
        sub_B99FD0(v61, v80, v79);
      }
      while ( v77 != v78 );
      i = v88;
      v7 = v84;
    }
LABEL_93:
    sub_BD84D0((__int64)v7, v61);
    v64 = (unsigned int)v119;
    v65 = (unsigned int)v119 + 1LL;
    if ( v65 > HIDWORD(v119) )
    {
      sub_C8D5F0((__int64)&v118, v120, v65, 8u, v62, v63);
      v64 = (unsigned int)v119;
    }
    v118[v64] = v7;
    LODWORD(v119) = v119 + 1;
    nullsub_61();
    v116 = &unk_49DA100;
    nullsub_63();
    if ( (_BYTE *)v101 != v103 )
      _libc_free(v101);
    if ( v96 > 0x40 )
    {
      v66 = v95;
      if ( v95 )
      {
LABEL_112:
        j_j___libc_free_0_0(v66);
        v93 = 1;
        goto LABEL_18;
      }
    }
LABEL_99:
    v93 = 1;
LABEL_18:
    for ( i = *(_QWORD *)(i + 8); i == v5 - 24 + 48; i = *(_QWORD *)(v5 + 32) )
    {
      v5 = *(_QWORD *)(v5 + 8);
      if ( v4 == v5 )
        goto LABEL_24;
      if ( !v5 )
        BUG();
    }
  }
  while ( v4 != v5 );
LABEL_24:
  v16 = v118;
  v17 = &v118[(unsigned int)v119];
  if ( v17 != v118 )
  {
    do
    {
      sub_F54ED0(*(v17 - 1));
      v18 = (__int64)*(v17 - 1);
      if ( (*(_BYTE *)(v18 + 7) & 0x40) != 0 )
      {
        v19 = *(_QWORD *)(v18 - 8);
        v18 = v19 + 32LL * (*(_DWORD *)(v18 + 4) & 0x7FFFFFF);
      }
      else
      {
        v19 = v18 - 32LL * (*(_DWORD *)(v18 + 4) & 0x7FFFFFF);
      }
      for ( ; v19 != v18; v19 += 32 )
      {
        if ( *(_QWORD *)v19 )
        {
          v20 = *(_QWORD *)(v19 + 8);
          **(_QWORD **)(v19 + 16) = v20;
          if ( v20 )
            *(_QWORD *)(v20 + 16) = *(_QWORD *)(v19 + 16);
        }
        *(_QWORD *)v19 = 0;
      }
      --v17;
    }
    while ( v17 != v16 );
    v16 = v118;
    v21 = &v118[(unsigned int)v119];
    if ( v21 != v118 )
    {
      do
      {
        v22 = *v16++;
        sub_B43D60(v22);
      }
      while ( v21 != v16 );
      v16 = v118;
    }
  }
  if ( v16 != (unsigned __int8 **)v120 )
    _libc_free((unsigned __int64)v16);
  return v93;
}
