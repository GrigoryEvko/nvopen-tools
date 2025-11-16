// Function: sub_24DBE00
// Address: 0x24dbe00
//
__int64 __fastcall sub_24DBE00(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v7; // rax
  const char *v8; // r13
  const char *v9; // r15
  __int64 v10; // r14
  _QWORD *v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rax
  __int64 v16; // rbx
  __int64 v17; // r12
  __int64 v18; // r14
  __int64 v19; // r13
  char *v20; // rbx
  char *v21; // r12
  __int64 v22; // rcx
  __int64 j; // rbx
  __int64 v24; // rax
  __int64 v25; // rax
  _QWORD *v26; // r15
  __int64 v27; // rdx
  unsigned int v28; // r8d
  _QWORD *v29; // rax
  __int64 v30; // r10
  _QWORD *v31; // rax
  _QWORD *v32; // r13
  __int64 v33; // rax
  __int64 v34; // rax
  char v35; // al
  char v36; // si
  _QWORD *v37; // rax
  __int64 v38; // r13
  __int64 v39; // rdx
  unsigned __int64 v40; // rbx
  unsigned __int64 v41; // r12
  __int64 v42; // rdx
  unsigned int v43; // esi
  unsigned __int8 *v44; // rax
  __int64 **v45; // rdi
  __int64 v46; // rax
  __int64 v47; // [rsp+18h] [rbp-268h]
  __int64 v48; // [rsp+20h] [rbp-260h]
  __int64 v49; // [rsp+28h] [rbp-258h]
  unsigned int v50; // [rsp+30h] [rbp-250h]
  __int64 v51; // [rsp+30h] [rbp-250h]
  __int64 v53; // [rsp+40h] [rbp-240h]
  __int64 v54; // [rsp+40h] [rbp-240h]
  __int64 v55; // [rsp+48h] [rbp-238h]
  __int64 v56; // [rsp+48h] [rbp-238h]
  __int64 i; // [rsp+50h] [rbp-230h]
  __int64 v58; // [rsp+58h] [rbp-228h]
  char v59; // [rsp+66h] [rbp-21Ah]
  unsigned __int8 v60; // [rsp+67h] [rbp-219h]
  __int64 v61; // [rsp+68h] [rbp-218h]
  char *v62; // [rsp+70h] [rbp-210h] BYREF
  char *v63; // [rsp+78h] [rbp-208h]
  char *v64; // [rsp+80h] [rbp-200h]
  __int64 v65; // [rsp+88h] [rbp-1F8h]
  __int64 v66; // [rsp+90h] [rbp-1F0h]
  _BYTE v67[32]; // [rsp+A0h] [rbp-1E0h] BYREF
  __int16 v68; // [rsp+C0h] [rbp-1C0h]
  __int64 v69; // [rsp+D0h] [rbp-1B0h] BYREF
  _QWORD *v70; // [rsp+D8h] [rbp-1A8h]
  int v71; // [rsp+E0h] [rbp-1A0h]
  int v72; // [rsp+E4h] [rbp-19Ch]
  int v73; // [rsp+E8h] [rbp-198h]
  char v74; // [rsp+ECh] [rbp-194h]
  _QWORD v75[3]; // [rsp+F0h] [rbp-190h] BYREF
  char *v76; // [rsp+108h] [rbp-178h]
  __int64 v77; // [rsp+110h] [rbp-170h]
  int v78; // [rsp+118h] [rbp-168h]
  char v79; // [rsp+11Ch] [rbp-164h]
  char v80; // [rsp+120h] [rbp-160h] BYREF
  _QWORD *v81; // [rsp+130h] [rbp-150h] BYREF
  unsigned __int64 v82; // [rsp+138h] [rbp-148h]
  char v83; // [rsp+14Ch] [rbp-134h]
  __int16 v84; // [rsp+150h] [rbp-130h]
  unsigned __int64 v85; // [rsp+168h] [rbp-118h]
  char v86; // [rsp+17Ch] [rbp-104h]
  const char *v87; // [rsp+190h] [rbp-F0h] BYREF
  __int64 v88; // [rsp+198h] [rbp-E8h]
  const char *v89; // [rsp+1A0h] [rbp-E0h]
  __int64 v90; // [rsp+1A8h] [rbp-D8h]
  const char *v91; // [rsp+1B0h] [rbp-D0h]
  unsigned __int64 v92; // [rsp+1B8h] [rbp-C8h] BYREF
  __int64 v93; // [rsp+1C0h] [rbp-C0h]
  _QWORD v94[4]; // [rsp+1C8h] [rbp-B8h] BYREF
  __int64 v95; // [rsp+1E8h] [rbp-98h]
  const char *v96; // [rsp+1F0h] [rbp-90h]
  __int64 v97; // [rsp+1F8h] [rbp-88h]
  __int64 *v98; // [rsp+200h] [rbp-80h]
  __int64 v99; // [rsp+208h] [rbp-78h]
  void **v100; // [rsp+210h] [rbp-70h]
  __int64 v101; // [rsp+218h] [rbp-68h]
  const char *v102; // [rsp+220h] [rbp-60h]
  __int64 v103; // [rsp+228h] [rbp-58h]
  const char *v104; // [rsp+230h] [rbp-50h]
  __int64 v105; // [rsp+238h] [rbp-48h] BYREF
  void *v106; // [rsp+240h] [rbp-40h] BYREF

  v87 = "llvm.coro.alloc";
  v89 = "llvm.coro.begin";
  v91 = "llvm.coro.subfn.addr";
  v93 = (__int64)"llvm.coro.free";
  v94[1] = "llvm.coro.id";
  v94[3] = "llvm.coro.id.retcon";
  v96 = "llvm.coro.id.async";
  v98 = (__int64 *)"llvm.coro.id.retcon.once";
  v100 = (void **)"llvm.coro.async.size.replace";
  v102 = "llvm.coro.async.resume";
  v88 = 15;
  v90 = 15;
  v92 = 20;
  v94[0] = 14;
  v94[2] = 12;
  v95 = 19;
  v97 = 18;
  v99 = 24;
  v101 = 28;
  v103 = 22;
  v104 = "llvm.coro.begin.custom.abi";
  v105 = 26;
  v60 = sub_24F32D0(a3, &v87, 11);
  if ( !v60 )
  {
    *(_QWORD *)(a1 + 48) = 0;
    *(_QWORD *)(a1 + 8) = a1 + 32;
    *(_QWORD *)(a1 + 56) = a1 + 80;
    *(_QWORD *)(a1 + 16) = 0x100000002LL;
    *(_QWORD *)(a1 + 64) = 2;
    *(_DWORD *)(a1 + 72) = 0;
    *(_BYTE *)(a1 + 76) = 1;
    *(_DWORD *)(a1 + 24) = 0;
    *(_BYTE *)(a1 + 28) = 1;
    *(_QWORD *)(a1 + 32) = &qword_4F82400;
    *(_QWORD *)a1 = 1;
    return a1;
  }
  v62 = 0;
  v7 = *(_QWORD *)(sub_BC0510(a4, &unk_4F82418, a3) + 8);
  v63 = 0;
  v64 = 0;
  v47 = v7;
  v65 = 0;
  v66 = 0;
  sub_29744A0(&v87);
  v8 = v89;
  v9 = v87;
  v10 = v88;
  v11 = (_QWORD *)sub_22077B0(0x20u);
  v12 = (__int64)v11;
  if ( v11 )
  {
    v11[1] = v9;
    v11[2] = v10;
    v11[3] = v8;
    *v11 = &unk_4A11BB8;
  }
  v81 = v11;
  if ( v63 == v64 )
  {
    sub_2353750((unsigned __int64 *)&v62, v63, &v81);
    v12 = (__int64)v81;
    goto LABEL_81;
  }
  if ( !v63 )
  {
    v63 = (char *)8;
LABEL_81:
    if ( v12 )
      (*(void (__fastcall **)(__int64))(*(_QWORD *)v12 + 8LL))(v12);
    goto LABEL_9;
  }
  *(_QWORD *)v63 = v11;
  v63 += 8;
LABEL_9:
  v71 = 2;
  v70 = v75;
  v76 = &v80;
  v73 = 0;
  v75[0] = &unk_4F82408;
  v74 = 1;
  v75[2] = 0;
  v77 = 2;
  v78 = 0;
  v79 = 1;
  v72 = 1;
  v69 = 1;
  sub_24F30B0(&v87, a3);
  BYTE6(v102) = 7;
  v101 = 0;
  v92 = (unsigned __int64)v94;
  v93 = 0x200000000LL;
  LODWORD(v102) = 0;
  v98 = (__int64 *)v88;
  v99 = (__int64)&v105;
  v100 = &v106;
  WORD2(v102) = 512;
  v103 = 0;
  v105 = (__int64)&unk_49DA100;
  v104 = 0;
  LOWORD(v97) = 0;
  v106 = &unk_49DA0B0;
  v13 = *(_QWORD *)(a3 + 32);
  v95 = 0;
  v96 = 0;
  v61 = v13;
  for ( i = a3 + 24; i != v61; v61 = *(_QWORD *)(v61 + 8) )
  {
    v14 = 49;
    v15 = 0;
    if ( v61 )
      v15 = v61 - 56;
    v58 = v15;
    v16 = v15;
    v59 = sub_B2D610(v15, 49);
    if ( v59 )
      v59 = (*(_BYTE *)(v16 + 32) & 0xFu) - 7 <= 1;
    v17 = *(_QWORD *)(v58 + 80);
    v18 = v58 + 72;
    if ( v58 + 72 != v17 )
    {
      if ( !v17 )
        BUG();
      while ( 1 )
      {
        v19 = *(_QWORD *)(v17 + 32);
        if ( v19 != v17 + 24 )
          break;
        v17 = *(_QWORD *)(v17 + 8);
        if ( v18 == v17 )
          goto LABEL_20;
        if ( !v17 )
          BUG();
      }
      if ( v17 != v18 )
      {
        v22 = 0;
LABEL_36:
        for ( j = *(_QWORD *)(v19 + 8); ; j = *(_QWORD *)(v17 + 32) )
        {
          v24 = v17 - 24;
          if ( !v17 )
            v24 = 0;
          if ( j != v24 + 48 )
          {
            if ( *(_BYTE *)(v19 - 24) == 85 )
            {
              v25 = *(_QWORD *)(v19 - 56);
              v26 = (_QWORD *)(v19 - 24);
              if ( v25 )
              {
                if ( !*(_BYTE *)v25 )
                {
LABEL_54:
                  if ( *(_QWORD *)(v25 + 24) == *(_QWORD *)(v19 + 56) && (*(_BYTE *)(v25 + 33) & 0x20) != 0 )
                  {
                    switch ( *(_DWORD *)(v25 + 36) )
                    {
                      case 0x1C:
                        v14 = sub_ACD6D0((__int64 *)v88);
                        sub_BD84D0((__int64)v26, v14);
                        goto LABEL_58;
                      case 0x22:
                        v14 = sub_AC9EC0(*(__int64 ***)(v19 - 16));
                        sub_BD84D0((__int64)v26, v14);
                        goto LABEL_58;
                      case 0x23:
                        v56 = *((_QWORD *)sub_BD3990(
                                            (unsigned __int8 *)v26[-4 * (*(_DWORD *)(v19 - 20) & 0x7FFFFFF)],
                                            v14)
                              - 4);
                        v44 = sub_BD3990((unsigned __int8 *)v26[4 * (1LL - (*(_DWORD *)(v19 - 20) & 0x7FFFFFF))], v14);
                        v14 = *(_QWORD *)(*((_QWORD *)v44 - 4)
                                        + 32 * (1LL - (*(_DWORD *)(*((_QWORD *)v44 - 4) + 4LL) & 0x7FFFFFF)));
                        if ( !(unsigned __int8)sub_AD8850(
                                                 *(_QWORD *)(v56 + 32 * (1LL - (*(_DWORD *)(v56 + 4) & 0x7FFFFFF))),
                                                 v14) )
                        {
                          v45 = *(__int64 ***)(v56 + 8);
                          v46 = *(_QWORD *)(v56 - 32LL * (*(_DWORD *)(v56 + 4) & 0x7FFFFFF));
                          v82 = v14;
                          v81 = (_QWORD *)v46;
                          v14 = sub_AD24A0(v45, (__int64 *)&v81, 2);
                          sub_BD84D0(v56, v14);
                        }
                        goto LABEL_58;
                      case 0x27:
                      case 0x28:
                      case 0x2F:
                        v14 = v26[4 * (1LL - (*(_DWORD *)(v19 - 20) & 0x7FFFFFF))];
                        sub_BD84D0((__int64)v26, v14);
                        goto LABEL_58;
                      case 0x2B:
                      case 0x3E:
                        if ( !v59 )
                          break;
                        v14 = sub_ACADE0(*(__int64 ***)(v19 - 16));
                        sub_BD84D0((__int64)v26, v14);
LABEL_58:
                        sub_B43D60(v26);
                        v22 = v60;
                        break;
                      case 0x30:
                      case 0x31:
                      case 0x32:
                      case 0x33:
                        v14 = sub_AC3540((__int64 *)v88);
                        sub_BD84D0((__int64)v26, v14);
                        goto LABEL_58;
                      case 0x3B:
                        sub_D5F1F0((__int64)&v92, (__int64)v26);
                        v27 = v26[4 * (1LL - (*(_DWORD *)(v19 - 20) & 0x7FFFFFF))];
                        v28 = *(_DWORD *)(v27 + 32);
                        v29 = *(_QWORD **)(v27 + 24);
                        if ( v28 > 0x40 )
                        {
                          v28 = *v29;
                          v30 = 8LL * v28;
                        }
                        else
                        {
                          v30 = 0;
                          if ( v28 )
                          {
                            v28 = (__int64)((_QWORD)v29 << (64 - (unsigned __int8)v28)) >> (64 - (unsigned __int8)v28);
                            v30 = 8LL * v28;
                          }
                        }
                        v48 = v30;
                        v49 = v26[-4 * (*(_DWORD *)(v19 - 20) & 0x7FFFFFF)];
                        v50 = v28;
                        v81 = (_QWORD *)sub_BCE3C0(v98, 0);
                        v82 = sub_BCE3C0(v98, 0);
                        v31 = (_QWORD *)sub_BD5C60((__int64)v26);
                        v32 = sub_BD0B90(v31, &v81, 2, 0);
                        sub_D5F1F0((__int64)&v92, (__int64)v26);
                        v84 = 257;
                        v33 = sub_24DBB60((__int64 *)&v92, (__int64)v32, v49, 0, v50, (__int64)&v81);
                        v68 = 257;
                        v51 = v33;
                        v53 = *(_QWORD *)(v32[2] + v48);
                        v34 = sub_AA4E30(v95);
                        v35 = sub_AE5020(v34, v53);
                        v84 = 257;
                        v36 = v35;
                        v37 = sub_BD2C40(80, unk_3F10A14);
                        v38 = (__int64)v37;
                        if ( v37 )
                          sub_B4D190((__int64)v37, v53, v51, (__int64)&v81, 0, v36, 0, 0);
                        (*((void (__fastcall **)(void **, __int64, _BYTE *, const char *, __int64))*v100 + 2))(
                          v100,
                          v38,
                          v67,
                          v96,
                          v97);
                        v39 = 16LL * (unsigned int)v93;
                        if ( v92 != v92 + v39 )
                        {
                          v55 = j;
                          v40 = v92 + v39;
                          v54 = v17;
                          v41 = v92;
                          do
                          {
                            v42 = *(_QWORD *)(v41 + 8);
                            v43 = *(_DWORD *)v41;
                            v41 += 16LL;
                            sub_B99FD0(v38, v43, v42);
                          }
                          while ( v40 != v41 );
                          j = v55;
                          v17 = v54;
                        }
                        v14 = v38;
                        sub_BD84D0((__int64)v26, v38);
                        goto LABEL_58;
                      default:
                        break;
                    }
                  }
                }
              }
            }
            if ( v18 == v17 )
              goto LABEL_47;
            v19 = j;
            goto LABEL_36;
          }
          v17 = *(_QWORD *)(v17 + 8);
          if ( v18 == v17 )
            break;
          if ( !v17 )
            BUG();
        }
        if ( *(_BYTE *)(v19 - 24) == 85 )
        {
          v25 = *(_QWORD *)(v19 - 56);
          v26 = (_QWORD *)(v19 - 24);
          if ( v25 )
          {
            if ( !*(_BYTE *)v25 )
              goto LABEL_54;
          }
        }
LABEL_47:
        if ( (_BYTE)v22 )
        {
          sub_BBE020(v47, v58, (__int64)&v69, v22);
          sub_BC2570((__int64)&v81, &v62, v58, v47);
          if ( !v86 )
            _libc_free(v85);
          if ( !v83 )
            _libc_free(v82);
        }
      }
    }
LABEL_20:
    ;
  }
  memset((void *)a1, 0, 0x60u);
  *(_DWORD *)(a1 + 16) = 2;
  *(_QWORD *)(a1 + 8) = a1 + 32;
  *(_QWORD *)(a1 + 56) = a1 + 80;
  *(_BYTE *)(a1 + 28) = 1;
  *(_DWORD *)(a1 + 64) = 2;
  *(_BYTE *)(a1 + 76) = 1;
  nullsub_61();
  v105 = (__int64)&unk_49DA100;
  nullsub_63();
  if ( (_QWORD *)v92 != v94 )
    _libc_free(v92);
  if ( !v79 )
    _libc_free((unsigned __int64)v76);
  if ( !v74 )
    _libc_free((unsigned __int64)v70);
  v20 = v63;
  v21 = v62;
  if ( v63 != v62 )
  {
    do
    {
      if ( *(_QWORD *)v21 )
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)v21 + 8LL))(*(_QWORD *)v21);
      v21 += 8;
    }
    while ( v20 != v21 );
    v21 = v62;
  }
  if ( v21 )
    j_j___libc_free_0((unsigned __int64)v21);
  return a1;
}
