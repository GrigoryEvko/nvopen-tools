// Function: sub_28E4750
// Address: 0x28e4750
//
__int64 __fastcall sub_28E4750(__int64 a1, __int64 a2, __int64 a3, unsigned __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r8
  __int64 v8; // rcx
  __int64 v9; // rdx
  _QWORD *v10; // r12
  _QWORD *v11; // r13
  __int64 v12; // r14
  _BYTE *v13; // rbx
  unsigned __int64 *v14; // r14
  unsigned __int8 v15; // r15
  _QWORD *v16; // rax
  _QWORD *v17; // r12
  _QWORD *v18; // r13
  __int64 v19; // r15
  __int64 v20; // r15
  __int64 v21; // rax
  __int64 v22; // r15
  __int64 v23; // rdi
  __int64 v24; // rax
  unsigned __int64 v25; // rax
  const char *v26; // rax
  const char *v27; // rdx
  __int64 v28; // rdi
  const char *v29; // rsi
  __int64 v30; // r9
  const char *v31; // r14
  __int64 v32; // rax
  int v33; // esi
  unsigned __int64 *v34; // rdx
  __int64 v36; // rdx
  __int64 v37; // rcx
  __int64 v38; // r8
  __int64 v39; // r9
  unsigned __int64 v40; // rbx
  _QWORD *v41; // r12
  void (__fastcall *v42)(_QWORD *, _QWORD *, __int64); // rax
  __int64 v43; // rax
  __int64 v44; // r15
  __int64 v45; // r14
  unsigned __int8 *v46; // rdx
  __int64 v47; // rax
  __int64 v48; // r14
  __int64 v49; // rdx
  unsigned __int64 *v50; // r12
  __int64 v51; // r13
  _BYTE *v52; // r14
  __int64 v53; // rbx
  __int64 v54; // rdx
  unsigned int v55; // esi
  __int64 v56; // rax
  __int64 v57; // rdx
  __int64 v58; // rdx
  int v59; // eax
  int v60; // eax
  unsigned int v61; // edx
  __int64 v62; // rax
  __int64 v63; // rdx
  __int64 v64; // rdx
  int v65; // eax
  int v66; // eax
  unsigned int v67; // edx
  __int64 v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rdx
  unsigned __int8 *v71; // rax
  unsigned __int64 v72; // r15
  _QWORD *v73; // [rsp+0h] [rbp-470h]
  __int64 v74; // [rsp+20h] [rbp-450h]
  __int64 v75; // [rsp+28h] [rbp-448h]
  __int64 v76; // [rsp+30h] [rbp-440h]
  unsigned __int8 v77; // [rsp+3Fh] [rbp-431h]
  unsigned __int64 v78; // [rsp+48h] [rbp-428h]
  unsigned __int8 *v79; // [rsp+48h] [rbp-428h]
  __int64 v80; // [rsp+50h] [rbp-420h]
  __int64 v81; // [rsp+58h] [rbp-418h]
  unsigned __int8 v83; // [rsp+70h] [rbp-400h]
  __int64 *v84; // [rsp+80h] [rbp-3F0h]
  __int64 v85; // [rsp+88h] [rbp-3E8h]
  __int64 v86; // [rsp+90h] [rbp-3E0h]
  __int64 v87; // [rsp+98h] [rbp-3D8h]
  __int64 v88; // [rsp+98h] [rbp-3D8h]
  unsigned int v89; // [rsp+B4h] [rbp-3BCh] BYREF
  unsigned int v90; // [rsp+B8h] [rbp-3B8h]
  int v91; // [rsp+BCh] [rbp-3B4h]
  const char *v92[4]; // [rsp+C0h] [rbp-3B0h] BYREF
  __int16 v93; // [rsp+E0h] [rbp-390h]
  unsigned __int64 *v94; // [rsp+F0h] [rbp-380h] BYREF
  __int64 v95; // [rsp+F8h] [rbp-378h]
  _BYTE v96[32]; // [rsp+100h] [rbp-370h] BYREF
  __int64 v97; // [rsp+120h] [rbp-350h]
  __int64 v98; // [rsp+128h] [rbp-348h]
  __int64 v99; // [rsp+130h] [rbp-340h]
  __int64 *v100; // [rsp+138h] [rbp-338h]
  void **v101; // [rsp+140h] [rbp-330h]
  void **v102; // [rsp+148h] [rbp-328h]
  __int64 v103; // [rsp+150h] [rbp-320h]
  int v104; // [rsp+158h] [rbp-318h]
  __int16 v105; // [rsp+15Ch] [rbp-314h]
  char v106; // [rsp+15Eh] [rbp-312h]
  __int64 v107; // [rsp+160h] [rbp-310h]
  __int64 v108; // [rsp+168h] [rbp-308h]
  void *v109; // [rsp+170h] [rbp-300h] BYREF
  void *v110; // [rsp+178h] [rbp-2F8h] BYREF
  unsigned __int64 v111[94]; // [rsp+180h] [rbp-2F0h] BYREF

  v6 = a1;
  memset(v111, 0, 0x2C0u);
  v8 = 0;
  v84 = (__int64 *)a2;
  if ( a4 )
  {
    v111[68] = a4;
    v111[0] = (unsigned __int64)&v111[2];
    v111[1] = 0x1000000000LL;
    v111[72] = (unsigned __int64)&v111[75];
    v111[66] = 0;
    v111[67] = 0;
    v111[69] = 0;
    LOBYTE(v111[70]) = 1;
    v111[71] = 0;
    v111[73] = 8;
    LODWORD(v111[74]) = 0;
    BYTE4(v111[74]) = 1;
    LOWORD(v111[83]) = 0;
    memset(&v111[84], 0, 24);
    LOBYTE(v111[87]) = 1;
  }
  v83 = 0;
  v87 = *(_QWORD *)(a1 + 80);
  v86 = a1 + 72;
  while ( 1 )
  {
    v9 = v87;
    if ( v86 == v87 )
      break;
    while ( 1 )
    {
      v8 = v87;
      v9 = *(_QWORD *)(v87 + 8);
      v10 = *(_QWORD **)(v87 + 32);
      v11 = (_QWORD *)(v87 + 24);
      v85 = v87;
      v87 = v9;
      if ( v10 != v11 )
        break;
LABEL_50:
      if ( v86 == v87 )
        goto LABEL_51;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        if ( !v10 )
          BUG();
        if ( *((_BYTE *)v10 - 24) != 85 )
          goto LABEL_7;
        v12 = *(v10 - 7);
        if ( !v12 || *(_BYTE *)v12 || *(_QWORD *)(v12 + 24) != v10[7] )
          goto LABEL_7;
        v13 = v10 - 3;
        if ( !(unsigned __int8)sub_A73ED0(v10 + 6, 23) && !(unsigned __int8)sub_B49560((__int64)(v10 - 3), 23) )
          break;
        if ( (unsigned __int8)sub_A73ED0(v10 + 6, 4) )
          break;
        a2 = 4;
        if ( (unsigned __int8)sub_B49560((__int64)(v10 - 3), 4) )
          break;
        v10 = (_QWORD *)v10[1];
        if ( v11 == v10 )
          goto LABEL_50;
      }
      a2 = 72;
      if ( !(unsigned __int8)sub_A73ED0(v10 + 6, 72) )
      {
        a2 = 72;
        if ( !(unsigned __int8)sub_B49560((__int64)(v10 - 3), 72)
          && (*((_WORD *)v10 - 11) & 3) != 2
          && (*(_BYTE *)(v12 + 32) & 0xFu) - 7 > 1 )
        {
          a2 = v12;
          if ( sub_981210(*v84, v12, &v89) )
          {
            v9 = (unsigned __int64)v89 >> 6;
            a2 = 1LL << v89;
            v8 = v84[v9 + 1] & (1LL << v89);
            v14 = (unsigned __int64 *)v8;
            if ( !v8 )
            {
              v8 = 2 * (v89 & 3);
              v9 = ((int)*(unsigned __int8 *)(*v84 + (v89 >> 2)) >> (2 * (v89 & 3))) & 3;
              if ( (((int)*(unsigned __int8 *)(*v84 + (v89 >> 2)) >> (2 * (v89 & 3))) & 3) != 0 && v89 - 448 <= 1 )
              {
                a2 = *(v10 - 2);
                v15 = sub_DFAEF0(a3);
                if ( v15 )
                {
                  if ( LOBYTE(v111[87]) )
                    v14 = v111;
                  if ( !(unsigned __int8)sub_B49E20((__int64)(v10 - 3)) )
                    break;
                }
              }
            }
          }
        }
      }
LABEL_7:
      v10 = (_QWORD *)v10[1];
      if ( v11 == v10 )
        goto LABEL_50;
    }
    v16 = v10;
    v17 = v11;
    v77 = v15;
    v18 = v16;
    v19 = v16[1];
    v80 = *(v16 - 2);
    if ( v19 == v16[2] + 48LL || !v19 )
      v20 = 0;
    else
      v20 = v19 - 24;
    v100 = (__int64 *)sub_BD5C60(v20);
    v101 = &v109;
    v102 = &v110;
    v105 = 512;
    LOWORD(v99) = 0;
    v94 = (unsigned __int64 *)v96;
    v109 = &unk_49DA100;
    v95 = 0x200000000LL;
    v110 = &unk_49DA0B0;
    v103 = 0;
    v104 = 0;
    v106 = 7;
    v107 = 0;
    v108 = 0;
    v97 = 0;
    v98 = 0;
    sub_D5F1F0((__int64)&v94, v20);
    v21 = v18[1];
    if ( v21 == v18[2] + 48LL || !v21 )
      v22 = 0;
    else
      v22 = v21 - 24;
    v23 = sub_ACD6D0(v100);
    v24 = v74;
    LOWORD(v24) = 0;
    v74 = v24;
    v78 = sub_F38250(v23, (__int64 *)(v22 + 24), v24, 0, 0, (__int64)v14, 0, 0);
    v25 = *(_QWORD *)(v85 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v17 == (_QWORD *)v25 )
      goto LABEL_107;
    if ( !v25 )
      BUG();
    v88 = v25 - 24;
    if ( (unsigned int)*(unsigned __int8 *)(v25 - 24) - 30 > 0xA )
    {
LABEL_107:
      v81 = -96;
      v88 = 0;
      sub_B4CC70(0);
    }
    else
    {
      v81 = v25 - 120;
      sub_B4CC70(v88);
    }
    v76 = sub_B46EC0(v78, 0);
    v26 = sub_BD5D20(v85 - 24);
    v93 = 773;
    v92[0] = v26;
    v92[2] = ".split";
    v92[1] = v27;
    sub_BD6B50((unsigned __int8 *)v76, v92);
    v28 = *(_QWORD *)(v76 + 56);
    v97 = v76;
    LOWORD(v99) = 1;
    v98 = v28;
    if ( v28 != v76 + 48 )
    {
      if ( v28 )
        v28 -= 24;
      v29 = *(const char **)sub_B46C60(v28);
      v92[0] = v29;
      if ( !v29 || (sub_B96E90((__int64)v92, (__int64)v29, 1), (v31 = v92[0]) == 0) )
      {
        sub_93FB40((__int64)&v94, 0);
        v31 = v92[0];
        goto LABEL_68;
      }
      v32 = (__int64)v94;
      v33 = v95;
      v34 = &v94[2 * (unsigned int)v95];
      if ( v94 != v34 )
      {
        while ( *(_DWORD *)v32 )
        {
          v32 += 16;
          if ( v34 == (unsigned __int64 *)v32 )
            goto LABEL_103;
        }
        *(const char **)(v32 + 8) = v92[0];
        goto LABEL_69;
      }
LABEL_103:
      if ( (unsigned int)v95 >= (unsigned __int64)HIDWORD(v95) )
      {
        v72 = v75 & 0xFFFFFFFF00000000LL;
        v75 &= 0xFFFFFFFF00000000LL;
        if ( HIDWORD(v95) < (unsigned __int64)(unsigned int)v95 + 1 )
        {
          sub_C8D5F0((__int64)&v94, v96, (unsigned int)v95 + 1LL, 0x10u, (unsigned int)v95 + 1LL, v30);
          v34 = &v94[2 * (unsigned int)v95];
        }
        *v34 = v72;
        v34[1] = (unsigned __int64)v31;
        v31 = v92[0];
        LODWORD(v95) = v95 + 1;
      }
      else
      {
        if ( v34 )
        {
          *(_DWORD *)v34 = 0;
          v34[1] = (unsigned __int64)v31;
          v33 = v95;
          v31 = v92[0];
        }
        LODWORD(v95) = v33 + 1;
      }
LABEL_68:
      if ( v31 )
LABEL_69:
        sub_B91220((__int64)v92, (__int64)v31);
    }
    v93 = 257;
    v44 = sub_D5C860((__int64 *)&v94, v80, 2, (__int64)v92);
    sub_BD84D0((__int64)v13, v44);
    v45 = v78;
    v46 = *(unsigned __int8 **)(v78 + 40);
    v92[0] = "call.sqrt";
    v79 = v46;
    v93 = 259;
    sub_BD6B50(v46, v92);
    sub_D5F1F0((__int64)&v94, v45);
    v47 = sub_B47F80(v13);
    v93 = 257;
    v48 = v47;
    (*((void (__fastcall **)(void **, __int64, const char **, __int64, __int64))*v102 + 2))(v102, v47, v92, v98, v99);
    v49 = 2LL * (unsigned int)v95;
    v50 = &v94[v49];
    if ( v94 != &v94[v49] )
    {
      v73 = v18;
      v51 = v48;
      v52 = v13;
      v53 = (__int64)v94;
      do
      {
        v54 = *(_QWORD *)(v53 + 8);
        v55 = *(_DWORD *)v53;
        v53 += 16;
        sub_B99FD0(v51, v55, v54);
      }
      while ( v50 != (unsigned __int64 *)v53 );
      v13 = v52;
      v48 = v51;
      v18 = v73;
    }
    sub_B49E10((__int64)v13);
    sub_D5F1F0((__int64)&v94, v88);
    if ( (unsigned __int8)sub_DFAF20(a3) )
    {
      v91 = 0;
      v93 = 257;
      v56 = sub_B35C90((__int64)&v94, 7u, (__int64)v13, (__int64)v13, (__int64)v92, 0, v90, 0);
    }
    else
    {
      v93 = 257;
      v71 = sub_AD8DD0(v80, 0.0);
      v91 = 0;
      v56 = sub_B35C90(
              (__int64)&v94,
              3u,
              *(_QWORD *)&v13[-32 * (*((_DWORD *)v18 - 5) & 0x7FFFFFF)],
              (__int64)v71,
              (__int64)v92,
              0,
              v90,
              0);
    }
    if ( *(_QWORD *)(v88 - 96) )
    {
      v57 = *(_QWORD *)(v88 - 88);
      **(_QWORD **)(v88 - 80) = v57;
      if ( v57 )
        *(_QWORD *)(v57 + 16) = *(_QWORD *)(v88 - 80);
    }
    a2 = v88;
    *(_QWORD *)(v88 - 96) = v56;
    if ( v56 )
    {
      v58 = *(_QWORD *)(v56 + 16);
      *(_QWORD *)(v88 - 88) = v58;
      if ( v58 )
      {
        a2 = v88 - 88;
        *(_QWORD *)(v58 + 16) = v88 - 88;
      }
      *(_QWORD *)(v88 - 80) = v56 + 16;
      *(_QWORD *)(v56 + 16) = v81;
    }
    v59 = *(_DWORD *)(v44 + 4) & 0x7FFFFFF;
    if ( v59 == *(_DWORD *)(v44 + 72) )
    {
      sub_B48D90(v44);
      v59 = *(_DWORD *)(v44 + 4) & 0x7FFFFFF;
    }
    v60 = (v59 + 1) & 0x7FFFFFF;
    v61 = v60 | *(_DWORD *)(v44 + 4) & 0xF8000000;
    v62 = *(_QWORD *)(v44 - 8) + 32LL * (unsigned int)(v60 - 1);
    *(_DWORD *)(v44 + 4) = v61;
    if ( *(_QWORD *)v62 )
    {
      v63 = *(_QWORD *)(v62 + 8);
      **(_QWORD **)(v62 + 16) = v63;
      if ( v63 )
        *(_QWORD *)(v63 + 16) = *(_QWORD *)(v62 + 16);
    }
    *(_QWORD *)v62 = v13;
    v64 = *(v18 - 1);
    *(_QWORD *)(v62 + 8) = v64;
    if ( v64 )
    {
      a2 = v62 + 8;
      *(_QWORD *)(v64 + 16) = v62 + 8;
    }
    *(_QWORD *)(v62 + 16) = v18 - 1;
    *(v18 - 1) = v62;
    *(_QWORD *)(*(_QWORD *)(v44 - 8)
              + 32LL * *(unsigned int *)(v44 + 72)
              + 8LL * ((*(_DWORD *)(v44 + 4) & 0x7FFFFFFu) - 1)) = v85 - 24;
    v65 = *(_DWORD *)(v44 + 4) & 0x7FFFFFF;
    if ( v65 == *(_DWORD *)(v44 + 72) )
    {
      sub_B48D90(v44);
      v65 = *(_DWORD *)(v44 + 4) & 0x7FFFFFF;
    }
    v66 = (v65 + 1) & 0x7FFFFFF;
    v67 = v66 | *(_DWORD *)(v44 + 4) & 0xF8000000;
    v68 = *(_QWORD *)(v44 - 8) + 32LL * (unsigned int)(v66 - 1);
    *(_DWORD *)(v44 + 4) = v67;
    if ( *(_QWORD *)v68 )
    {
      v69 = *(_QWORD *)(v68 + 8);
      **(_QWORD **)(v68 + 16) = v69;
      if ( v69 )
        *(_QWORD *)(v69 + 16) = *(_QWORD *)(v68 + 16);
    }
    *(_QWORD *)v68 = v48;
    if ( v48 )
    {
      v70 = *(_QWORD *)(v48 + 16);
      *(_QWORD *)(v68 + 8) = v70;
      if ( v70 )
      {
        a2 = v68 + 8;
        *(_QWORD *)(v70 + 16) = v68 + 8;
      }
      *(_QWORD *)(v68 + 16) = v48 + 16;
      *(_QWORD *)(v48 + 16) = v68;
    }
    *(_QWORD *)(*(_QWORD *)(v44 - 8)
              + 32LL * *(unsigned int *)(v44 + 72)
              + 8LL * ((*(_DWORD *)(v44 + 4) & 0x7FFFFFFu) - 1)) = v79;
    v87 = v76 + 24;
    nullsub_61();
    v109 = &unk_49DA100;
    nullsub_63();
    if ( v94 != (unsigned __int64 *)v96 )
      _libc_free((unsigned __int64)v94);
    v83 = v77;
  }
LABEL_51:
  if ( LOBYTE(v111[87]) )
  {
    LOBYTE(v111[87]) = 0;
    sub_FFCE90((__int64)v111, a2, v9, v8, v6, a6);
    sub_FFD870((__int64)v111, a2, v36, v37, v38, v39);
    sub_FFBC40((__int64)v111, a2);
    v40 = v111[85];
    v41 = (_QWORD *)v111[84];
    if ( v111[85] != v111[84] )
    {
      do
      {
        v42 = (void (__fastcall *)(_QWORD *, _QWORD *, __int64))v41[7];
        *v41 = &unk_49E5048;
        if ( v42 )
          v42(v41 + 5, v41 + 5, 3);
        *v41 = &unk_49DB368;
        v43 = v41[3];
        if ( v43 != -4096 && v43 != 0 && v43 != -8192 )
          sub_BD60C0(v41 + 1);
        v41 += 9;
      }
      while ( (_QWORD *)v40 != v41 );
      v41 = (_QWORD *)v111[84];
    }
    if ( v41 )
      j_j___libc_free_0((unsigned __int64)v41);
    if ( !BYTE4(v111[74]) )
      _libc_free(v111[72]);
    if ( (unsigned __int64 *)v111[0] != &v111[2] )
      _libc_free(v111[0]);
  }
  return v83;
}
