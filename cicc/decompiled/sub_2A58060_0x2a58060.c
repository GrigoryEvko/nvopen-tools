// Function: sub_2A58060
// Address: 0x2a58060
//
__int64 __fastcall sub_2A58060(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // r13
  __int64 v8; // r14
  __int64 v9; // rax
  char v10; // r15
  __int64 v11; // r12
  __int64 v12; // rax
  void *v13; // rsi
  void *v14; // r12
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // rbx
  __int64 v21; // rsi
  __int64 v22; // r13
  __int64 *v23; // rax
  __int64 *v24; // r13
  __int64 *v25; // rbx
  __int64 v26; // rdi
  __int64 v27; // rcx
  __int64 v28; // rbx
  __int64 v29; // r14
  __int64 *v30; // rbx
  unsigned __int64 v31; // r13
  __int64 *v32; // rax
  __int64 *v33; // rax
  unsigned __int64 v34; // rdx
  __int16 v35; // cx
  char v36; // r15
  _QWORD *v37; // rax
  __int64 v38; // r8
  __int64 v39; // r9
  char v40; // r15
  __int64 v41; // r12
  __int64 *v42; // rax
  __int64 *v43; // rdx
  __int64 *i; // rdx
  __int64 v45; // rax
  unsigned __int64 *v46; // r8
  __int64 v47; // r15
  unsigned __int64 *v48; // r13
  unsigned __int64 v49; // r11
  __int64 **v50; // r14
  unsigned __int8 *v51; // r12
  __int64 v52; // rax
  unsigned __int64 v53; // r14
  __int64 **v54; // rax
  __int64 v55; // rax
  __int64 v56; // rax
  __int64 v57; // r14
  __int64 v58; // rdi
  unsigned __int8 *v59; // r13
  __int64 (__fastcall *v60)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char); // rax
  __int64 v61; // rbx
  unsigned __int64 v62; // rsi
  __int64 v63; // rdx
  __int64 v64; // rax
  __int64 v65; // r13
  unsigned int *v66; // r13
  unsigned int *v67; // r12
  __int64 v68; // rdx
  unsigned int v69; // esi
  char v71; // [rsp+Fh] [rbp-391h]
  __int64 *v73; // [rsp+30h] [rbp-370h]
  __int64 **v74; // [rsp+40h] [rbp-360h]
  _QWORD *v75; // [rsp+48h] [rbp-358h]
  __int64 v76; // [rsp+50h] [rbp-350h]
  __int64 v77; // [rsp+70h] [rbp-330h]
  __int64 v78; // [rsp+78h] [rbp-328h]
  __int64 v79; // [rsp+80h] [rbp-320h]
  char v80; // [rsp+88h] [rbp-318h]
  unsigned __int64 v81; // [rsp+88h] [rbp-318h]
  char v82; // [rsp+90h] [rbp-310h]
  unsigned __int64 *v83; // [rsp+90h] [rbp-310h]
  __int16 v84; // [rsp+98h] [rbp-308h]
  unsigned __int64 v85; // [rsp+98h] [rbp-308h]
  __int64 v86; // [rsp+98h] [rbp-308h]
  __int64 v87[4]; // [rsp+A0h] [rbp-300h] BYREF
  char v88; // [rsp+C0h] [rbp-2E0h]
  char v89; // [rsp+C1h] [rbp-2DFh]
  unsigned int *v90; // [rsp+D0h] [rbp-2D0h] BYREF
  __int64 v91; // [rsp+D8h] [rbp-2C8h]
  _BYTE v92[32]; // [rsp+E0h] [rbp-2C0h] BYREF
  __int64 v93; // [rsp+100h] [rbp-2A0h]
  __int64 v94; // [rsp+108h] [rbp-298h]
  __int64 v95; // [rsp+110h] [rbp-290h]
  __int64 v96; // [rsp+118h] [rbp-288h]
  void **v97; // [rsp+120h] [rbp-280h]
  void **v98; // [rsp+128h] [rbp-278h]
  __int64 v99; // [rsp+130h] [rbp-270h]
  int v100; // [rsp+138h] [rbp-268h]
  __int16 v101; // [rsp+13Ch] [rbp-264h]
  char v102; // [rsp+13Eh] [rbp-262h]
  __int64 v103; // [rsp+140h] [rbp-260h]
  __int64 v104; // [rsp+148h] [rbp-258h]
  void *v105; // [rsp+150h] [rbp-250h] BYREF
  void *v106; // [rsp+158h] [rbp-248h] BYREF
  unsigned __int64 v107; // [rsp+160h] [rbp-240h] BYREF
  unsigned __int64 v108; // [rsp+168h] [rbp-238h]
  __int64 v109; // [rsp+170h] [rbp-230h] BYREF
  int v110; // [rsp+178h] [rbp-228h]
  char v111; // [rsp+17Ch] [rbp-224h]
  _QWORD v112[2]; // [rsp+180h] [rbp-220h] BYREF
  __int64 v113; // [rsp+190h] [rbp-210h] BYREF
  _BYTE *v114; // [rsp+198h] [rbp-208h]
  __int64 v115; // [rsp+1A0h] [rbp-200h]
  int v116; // [rsp+1A8h] [rbp-1F8h]
  char v117; // [rsp+1ACh] [rbp-1F4h]
  _BYTE v118[496]; // [rsp+1B0h] [rbp-1F0h] BYREF

  v4 = a3 + 24;
  v5 = sub_BC0510(a4, &unk_4F82418, a3);
  v6 = *(_QWORD *)(v4 + 8);
  v7 = *(_QWORD *)(v5 + 8);
  if ( v6 != v4 )
  {
    while ( 1 )
    {
      v8 = v6 - 56;
      if ( !v6 )
        v8 = 0;
      if ( !sub_B2FC80(v8) )
        break;
      v6 = *(_QWORD *)(v6 + 8);
      if ( v4 == v6 )
        goto LABEL_6;
    }
    v9 = sub_BC1CD0(v7, &unk_4F89C30, v8);
    if ( !(unsigned __int8)sub_DFA9B0(v9 + 8) )
      goto LABEL_41;
  }
LABEL_6:
  v10 = 0;
  v78 = *(_QWORD *)(a3 + 16);
  if ( a3 + 8 == v78 )
  {
LABEL_41:
    v13 = (void *)(a1 + 32);
    v14 = (void *)(a1 + 80);
    goto LABEL_42;
  }
  do
  {
    v11 = v78;
    v78 = *(_QWORD *)(v78 + 8);
    v77 = v11 - 56;
    if ( sub_B2FC80(v11 - 56) )
      continue;
    if ( (*(_BYTE *)(v11 + 24) & 1) == 0 )
      continue;
    v12 = *(_QWORD *)(v11 - 40);
    if ( !v12 )
      continue;
    if ( *(_QWORD *)(v12 + 8) )
      continue;
    v16 = *(_QWORD *)(v12 + 24);
    if ( *(_BYTE *)v16 != 63 )
      continue;
    v17 = *(_QWORD *)(v16 + 16);
    if ( !v17 )
      continue;
    if ( *(_QWORD *)(v17 + 8) )
      continue;
    if ( *(_QWORD *)(v11 - 32) != *(_QWORD *)(v16 + 72) )
      continue;
    v18 = *(_QWORD *)(v17 + 24);
    if ( *(_BYTE *)v18 != 61 )
      continue;
    v19 = *(_QWORD *)(v18 + 16);
    if ( !v19 )
      continue;
    if ( *(_QWORD *)(v19 + 8) )
      continue;
    if ( *(_QWORD *)(v18 + 8) != *(_QWORD *)(v16 + 80) )
      continue;
    if ( (*(_BYTE *)(v11 - 24) & 0xFu) - 7 > 1 )
      continue;
    if ( (*(_BYTE *)(v11 - 23) & 0x40) == 0 )
      continue;
    v71 = sub_2624ED0(v77);
    if ( !v71 )
      continue;
    v20 = *(_QWORD *)(v11 - 88);
    if ( *(_BYTE *)v20 != 9 )
      continue;
    v21 = *(_QWORD *)(*(_QWORD *)(v20 + 8) + 24LL);
    if ( *(_BYTE *)(v21 + 8) != 14 || (unsigned int)sub_AE43A0(a3 + 312, v21) != 64 )
      continue;
    v22 = 4LL * (*(_DWORD *)(v20 + 4) & 0x7FFFFFF);
    v23 = (__int64 *)(v20 - v22 * 8);
    if ( (*(_BYTE *)(v20 + 7) & 0x40) != 0 )
      v23 = *(__int64 **)(v20 - 8);
    v24 = &v23[v22];
    v25 = v23;
    while ( v24 != v25 )
    {
      v26 = *v25;
      LODWORD(v108) = 1;
      v107 = 0;
      if ( !(unsigned __int8)sub_96E080(v26, &v90, (__int64)&v107, a3 + 312, 0)
        || *(_BYTE *)v90 != 3
        || (v90[20] & 1) == 0
        || (v90[8] & 0xF) - 7 > 1
        || (*((_BYTE *)v90 + 33) & 0x40) == 0
        || (v25 += 4, !(unsigned __int8)sub_2624ED0((__int64)v90)) )
      {
        sub_969240((__int64 *)&v107);
        goto LABEL_10;
      }
      sub_969240((__int64 *)&v107);
    }
    v27 = *(_QWORD *)(*(_QWORD *)(v11 - 40) + 24LL);
    v28 = *(_QWORD *)(v27 + 40);
    v76 = v27;
    v75 = *(_QWORD **)(*(_QWORD *)(v27 + 16) + 24LL);
    v73 = *(__int64 **)(v11 - 16);
    v96 = sub_AA48A0(v28);
    v97 = &v105;
    v98 = &v106;
    v90 = (unsigned int *)v92;
    v105 = &unk_49DA100;
    v91 = 0x200000000LL;
    LOWORD(v95) = 0;
    v106 = &unk_49DA0B0;
    v99 = 0;
    v100 = 0;
    v101 = 512;
    v102 = 7;
    v103 = 0;
    v104 = 0;
    v93 = v28;
    v94 = v28 + 48;
    v29 = *(_QWORD *)(v11 - 88);
    v30 = *(__int64 **)(*(_QWORD *)(v28 + 72) + 40LL);
    v31 = (unsigned int)*(_QWORD *)(*(_QWORD *)(v29 + 8) + 32LL);
    v79 = *(_QWORD *)(*(_QWORD *)(v29 + 8) + 32LL);
    v32 = (__int64 *)sub_BCB2D0((_QWORD *)*v30);
    v74 = (__int64 **)sub_BCD420(v32, v31);
    v80 = *(_BYTE *)(v11 + 24) & 1;
    v82 = *(_BYTE *)(v11 - 24) & 0xF;
    v33 = (__int64 *)sub_BD5D20(v77);
    v109 = (__int64)".rel";
    LOWORD(v112[0]) = 773;
    v107 = (unsigned __int64)v33;
    v108 = v34;
    v35 = (*(_BYTE *)(v11 - 23) >> 2) & 7;
    LODWORD(v33) = *(_DWORD *)(*(_QWORD *)(v11 - 48) + 8LL);
    BYTE4(v87[0]) = 1;
    v84 = v35;
    LODWORD(v87[0]) = (unsigned int)v33 >> 8;
    v36 = *(_BYTE *)(v11 + 24) >> 1;
    v37 = sub_BD2C40(88, unk_3F0FAE8);
    v40 = v36 & 1;
    v41 = (__int64)v37;
    if ( v37 )
      sub_B30000((__int64)v37, (__int64)v30, v74, v80, v82, 0, (__int64)&v107, v77, v84, v87[0], v40);
    v42 = &v109;
    v43 = &v109;
    v107 = (unsigned __int64)&v109;
    v108 = 0x4000000000LL;
    if ( v31 )
    {
      if ( v31 > 0x40 )
      {
        sub_C8D5F0((__int64)&v107, &v109, v31, 8u, v38, v39);
        v43 = (__int64 *)v107;
        v42 = (__int64 *)(v107 + 8LL * (unsigned int)v108);
      }
      for ( i = &v43[v31]; i != v42; ++v42 )
      {
        if ( v42 )
          *v42 = 0;
      }
      LODWORD(v108) = v79;
    }
    v45 = 4LL * (*(_DWORD *)(v29 + 4) & 0x7FFFFFF);
    if ( (*(_BYTE *)(v29 + 7) & 0x40) != 0 )
    {
      v46 = *(unsigned __int64 **)(v29 - 8);
      v83 = &v46[v45];
    }
    else
    {
      v83 = (unsigned __int64 *)v29;
      v46 = (unsigned __int64 *)(v29 - v45 * 8);
    }
    if ( v46 != v83 )
    {
      v81 = v41;
      v47 = 0;
      v48 = v46;
      do
      {
        v49 = *v48;
        v48 += 4;
        v85 = v49;
        v50 = (__int64 **)sub_AE4420((__int64)(v30 + 39), *v30, 0);
        v51 = (unsigned __int8 *)sub_AD4C50(v81, v50, 0);
        v52 = sub_AD4C50(v85, v50, 0);
        v53 = sub_AD57F0(v52, v51, 0, 0);
        v54 = (__int64 **)sub_BCB2D0((_QWORD *)*v30);
        v55 = sub_AD4C30(v53, v54, 0);
        *(_QWORD *)(v107 + v47) = v55;
        v47 += 8;
      }
      while ( v83 != v48 );
      v41 = v81;
    }
    v56 = sub_AD1300(v74, (__int64 *)v107, (unsigned int)v108);
    sub_B30160(v41, v56);
    *(_BYTE *)(v41 + 32) = *(_BYTE *)(v41 + 32) & 0x3F | 0x80;
    sub_B2F770(v41, 2u);
    if ( (__int64 *)v107 != &v109 )
      _libc_free(v107);
    sub_D5F1F0((__int64)&v90, v76);
    v57 = *(_QWORD *)(v76 + 32 * (2LL - (*(_DWORD *)(v76 + 4) & 0x7FFFFFF)));
    v58 = *(_QWORD *)(v57 + 8);
    v89 = 1;
    v87[0] = (__int64)"reltable.shift";
    v88 = 3;
    v59 = (unsigned __int8 *)sub_ACD640(v58, 2, 0);
    v60 = (__int64 (__fastcall *)(__int64, unsigned int, _BYTE *, unsigned __int8 *, unsigned __int8, char))*((_QWORD *)*v97 + 4);
    if ( v60 == sub_9201A0 )
    {
      if ( *(_BYTE *)v57 > 0x15u || *v59 > 0x15u )
      {
LABEL_78:
        LOWORD(v112[0]) = 257;
        v61 = sub_B504D0(25, v57, (__int64)v59, (__int64)&v107, 0, 0);
        (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v98 + 2))(v98, v61, v87, v94, v95);
        v65 = 4LL * (unsigned int)v91;
        if ( v90 != &v90[v65] )
        {
          v86 = v41;
          v66 = &v90[v65];
          v67 = v90;
          do
          {
            v68 = *((_QWORD *)v67 + 1);
            v69 = *v67;
            v67 += 4;
            sub_B99FD0(v61, v69, v68);
          }
          while ( v66 != v67 );
          v41 = v86;
        }
        goto LABEL_72;
      }
      if ( (unsigned __int8)sub_AC47B0(25) )
        v61 = sub_AD5570(25, v57, v59, 0, 0);
      else
        v61 = sub_AABE40(0x19u, (unsigned __int8 *)v57, v59);
    }
    else
    {
      v61 = v60((__int64)v97, 25u, (_BYTE *)v57, v59, 0, 0);
    }
    if ( !v61 )
      goto LABEL_78;
LABEL_72:
    sub_D5F1F0((__int64)&v90, (__int64)v75);
    v107 = *(_QWORD *)(v57 + 8);
    v62 = 0;
    v63 = sub_B6E160(v73, 0xD6u, (__int64)&v107, 1);
    LOWORD(v112[0]) = 259;
    v107 = (unsigned __int64)"reltable.intrinsic";
    v87[0] = v41;
    v87[1] = v61;
    if ( v63 )
      v62 = *(_QWORD *)(v63 + 24);
    v64 = sub_921880(&v90, v62, v63, (int)v87, 2, (__int64)&v107, 0);
    sub_BD84D0((__int64)v75, v64);
    sub_B43D60(v75);
    sub_B43D60((_QWORD *)v76);
    nullsub_61();
    v105 = &unk_49DA100;
    nullsub_63();
    if ( v90 != (unsigned int *)v92 )
      _libc_free((unsigned __int64)v90);
    sub_B30290(v77);
    v10 = v71;
LABEL_10:
    ;
  }
  while ( a3 + 8 != v78 );
  v13 = (void *)(a1 + 32);
  v14 = (void *)(a1 + 80);
  if ( !v10 )
  {
LABEL_42:
    *(_QWORD *)(a1 + 8) = v13;
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 56) = v14;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v108 = (unsigned __int64)v112;
  v112[0] = &unk_4F82408;
  v109 = 0x100000002LL;
  v110 = 0;
  v111 = 1;
  v113 = 0;
  v114 = v118;
  v115 = 2;
  v116 = 0;
  v117 = 1;
  v107 = 1;
  sub_C8CF70(a1, v13, 2, (__int64)v112, (__int64)&v107);
  sub_C8CF70(a1 + 48, v14, 2, (__int64)v118, (__int64)&v113);
  if ( !v117 )
  {
    _libc_free((unsigned __int64)v114);
    if ( v111 )
      return a1;
    goto LABEL_45;
  }
  if ( !v111 )
LABEL_45:
    _libc_free(v108);
  return a1;
}
