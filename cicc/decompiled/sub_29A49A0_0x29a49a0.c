// Function: sub_29A49A0
// Address: 0x29a49a0
//
_QWORD *__fastcall sub_29A49A0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  __int64 v6; // r14
  __int64 *v7; // rsi
  __int64 v8; // r15
  unsigned __int8 *v9; // rdi
  unsigned __int8 *v10; // r13
  _QWORD *v11; // r13
  __int64 v12; // rbx
  _QWORD *v13; // rax
  __int64 v14; // rbx
  _BYTE *v15; // rbx
  __int64 v16; // r14
  _BYTE *v17; // r12
  __int64 v18; // rdx
  unsigned int v19; // esi
  __int64 v20; // rax
  __int64 v21; // rdx
  __int64 v22; // rdi
  __int64 v23; // rsi
  int v24; // edx
  _QWORD *v25; // rax
  __int64 v26; // rax
  __int64 v27; // rax
  __int64 v28; // rdx
  __int64 v29; // r14
  __int64 v30; // rbx
  __int64 v31; // rdi
  __int64 v32; // rdx
  unsigned __int8 **v33; // rax
  __int64 v34; // rdx
  int v35; // eax
  int v36; // eax
  unsigned int v37; // ecx
  __int64 v38; // rax
  __int64 v39; // rcx
  __int64 v40; // rcx
  __int64 v41; // rax
  __int64 v42; // rax
  __int64 v43; // rax
  __int64 v44; // rdx
  __int64 v45; // rax
  __int64 v46; // rax
  __int64 v47; // r14
  __int64 v48; // rdi
  const char *v49; // rsi
  __int64 v50; // r8
  __int64 v51; // r9
  const char *v52; // r14
  _BYTE *v53; // rax
  unsigned int v54; // ecx
  _QWORD *v55; // rdx
  __int64 v56; // rax
  __int64 v57; // rbx
  __int64 v58; // r15
  __int64 v59; // r8
  __int64 v60; // r9
  _BYTE *v61; // r15
  _BYTE *v62; // r14
  __int64 v63; // rdx
  unsigned int v64; // esi
  __int64 v65; // r15
  signed __int64 v66; // r14
  __int64 v67; // rdx
  const char *v68; // rdx
  const char *v69; // r15
  __int64 *v70; // r14
  __int64 v71; // rdi
  __int64 v72; // r15
  int v73; // eax
  int v74; // eax
  unsigned int v75; // edx
  __int64 v76; // rax
  __int64 v77; // rdx
  __int64 v78; // rdx
  __int64 v79; // r12
  int v80; // eax
  int v81; // eax
  unsigned int v82; // edx
  __int64 v83; // rax
  __int64 v84; // rdx
  __int64 v85; // rdx
  unsigned __int64 v87; // rax
  unsigned __int8 *v88; // rdi
  _QWORD *v89; // r14
  __int64 v90; // rbx
  __int64 v91; // rdx
  _QWORD *v92; // r8
  __int64 v93; // rbx
  _QWORD *v94; // r12
  __int64 v95; // rsi
  int v96; // r14d
  __int64 v97; // rax
  __int64 v98; // [rsp+0h] [rbp-1D0h]
  unsigned __int8 *v99; // [rsp+8h] [rbp-1C8h]
  unsigned __int8 *v100; // [rsp+10h] [rbp-1C0h]
  _QWORD *v101; // [rsp+28h] [rbp-1A8h]
  __int64 v102; // [rsp+28h] [rbp-1A8h]
  __int64 v103; // [rsp+28h] [rbp-1A8h]
  _QWORD *v104; // [rsp+28h] [rbp-1A8h]
  __int64 v105; // [rsp+28h] [rbp-1A8h]
  _QWORD *v106; // [rsp+40h] [rbp-190h] BYREF
  _QWORD *v107; // [rsp+48h] [rbp-188h] BYREF
  char v108[32]; // [rsp+50h] [rbp-180h] BYREF
  __int16 v109; // [rsp+70h] [rbp-160h]
  _BYTE *v110; // [rsp+80h] [rbp-150h] BYREF
  unsigned int v111; // [rsp+88h] [rbp-148h]
  unsigned int v112; // [rsp+8Ch] [rbp-144h]
  _BYTE v113[32]; // [rsp+90h] [rbp-140h] BYREF
  __int64 v114; // [rsp+B0h] [rbp-120h]
  __int64 v115; // [rsp+B8h] [rbp-118h]
  __int64 v116; // [rsp+C0h] [rbp-110h]
  __int64 v117; // [rsp+D8h] [rbp-F8h]
  __int64 v118; // [rsp+E0h] [rbp-F0h]
  int v119; // [rsp+E8h] [rbp-E8h]
  void *v120; // [rsp+100h] [rbp-D0h]
  char *v121; // [rsp+110h] [rbp-C0h] BYREF
  __int64 v122; // [rsp+118h] [rbp-B8h]
  _BYTE v123[16]; // [rsp+120h] [rbp-B0h] BYREF
  __int16 v124; // [rsp+130h] [rbp-A0h]

  v4 = a1;
  sub_23D0AB0((__int64)&v110, a1, 0, 0, 0);
  v6 = *(_QWORD *)(a1 + 40);
  v7 = (__int64 *)(a1 + 24);
  if ( sub_B49200(a1) )
  {
    v87 = sub_F38250(a2, v7, 0, 0, a3, 0, 0, 0);
    v88 = *(unsigned __int8 **)(v87 + 40);
    v89 = (_QWORD *)v87;
    v121 = "if.true.direct_targ";
    v124 = 259;
    sub_BD6B50(v88, (const char **)&v121);
    v11 = (_QWORD *)sub_B47F80((_BYTE *)v4);
    sub_B44220(v11, (__int64)(v89 + 3), 0);
    v90 = *(_QWORD *)(v4 + 32);
    if ( v90 != *(_QWORD *)(v4 + 40) + 48LL && v90 )
    {
      if ( *(_BYTE *)(v90 - 24) == 78 )
      {
        v104 = (_QWORD *)sub_B47F80((_BYTE *)(v90 - 24));
        sub_BD2ED0((__int64)v104, v4, (__int64)v11);
        sub_B44220(v104, (__int64)(v89 + 3), 0);
        v91 = *(_QWORD *)(v90 + 8);
        if ( v91 == *(_QWORD *)(v90 + 16) + 48LL || !v91 )
          goto LABEL_129;
        v92 = v104;
        v93 = v91 - 24;
      }
      else
      {
        v93 = v90 - 24;
        v92 = v11;
      }
      if ( *(_BYTE *)v93 == 30 )
      {
        v105 = (__int64)v92;
        v94 = (_QWORD *)sub_B47F80((_BYTE *)v93);
        if ( (*(_DWORD *)(v93 + 4) & 0x7FFFFFF) != 0 )
        {
          v95 = *(_QWORD *)(v93 - 32LL * (*(_DWORD *)(v93 + 4) & 0x7FFFFFF));
          if ( v95 )
            sub_BD2ED0((__int64)v94, v95, v105);
        }
        sub_B44220(v94, (__int64)(v89 + 3), 0);
        sub_B43D60(v89);
        goto LABEL_92;
      }
    }
LABEL_129:
    sub_B47F80(0);
    BUG();
  }
  v106 = 0;
  v107 = 0;
  sub_F38330(a2, v7, 0, (unsigned __int64 *)&v106, (unsigned __int64 *)&v107, a3, 0, 0);
  v8 = *(_QWORD *)(a1 + 40);
  v9 = (unsigned __int8 *)v106[5];
  v10 = (unsigned __int8 *)v107[5];
  v99 = v9;
  v121 = "if.true.direct_targ";
  v100 = v10;
  v124 = 259;
  sub_BD6B50(v9, (const char **)&v121);
  v121 = "if.false.orig_indirect";
  v124 = 259;
  sub_BD6B50(v10, (const char **)&v121);
  v121 = "if.end.icp";
  v124 = 259;
  sub_BD6B50((unsigned __int8 *)v8, (const char **)&v121);
  v11 = (_QWORD *)sub_B47F80((_BYTE *)v4);
  sub_B444E0((_QWORD *)v4, (__int64)(v107 + 3), 0);
  sub_B44220(v11, (__int64)(v106 + 3), 0);
  if ( *(_BYTE *)v4 == 34 )
  {
    sub_B43D60(v106);
    sub_B43D60(v107);
    v114 = v8;
    v12 = *(_QWORD *)(v4 - 96);
    v115 = v8 + 48;
    LOWORD(v116) = 0;
    v124 = 257;
    v13 = sub_BD2C40(72, 1u);
    v101 = v13;
    if ( v13 )
      sub_B4C8F0((__int64)v13, v12, 1u, 0, 0);
    (*(void (__fastcall **)(__int64, _QWORD *, char **, __int64, __int64))(*(_QWORD *)v117 + 16LL))(
      v117,
      v101,
      &v121,
      v115,
      v116);
    v14 = 16LL * v111;
    if ( v110 != &v110[v14] )
    {
      v98 = v6;
      v15 = &v110[v14];
      v16 = (__int64)v101;
      v102 = v4;
      v17 = v110;
      do
      {
        v18 = *((_QWORD *)v17 + 1);
        v19 = *(_DWORD *)v17;
        v17 += 16;
        sub_B99FD0(v16, v19, v18);
      }
      while ( v15 != v17 );
      v6 = v98;
      v4 = v102;
    }
    v20 = sub_AA5930(*(_QWORD *)(v4 - 96));
    v22 = v21;
    v23 = v20;
    while ( v22 != v23 )
    {
      if ( (*(_DWORD *)(v23 + 4) & 0x7FFFFFF) != 0 )
      {
        v24 = 0;
        v25 = (_QWORD *)(*(_QWORD *)(v23 - 8) + 32LL * *(unsigned int *)(v23 + 72));
        while ( v6 != *v25 )
        {
          ++v24;
          ++v25;
          if ( (*(_DWORD *)(v23 + 4) & 0x7FFFFFF) == v24 )
            goto LABEL_15;
        }
        *v25 = v8;
      }
LABEL_15:
      v26 = *(_QWORD *)(v23 + 32);
      if ( !v26 )
        BUG();
      v23 = 0;
      if ( *(_BYTE *)(v26 - 24) == 84 )
        v23 = v26 - 24;
    }
    v27 = sub_AA5930(*(_QWORD *)(v4 - 64));
    v29 = v28;
    v30 = v27;
    while ( v29 != v30 )
    {
      if ( (*(_DWORD *)(v30 + 4) & 0x7FFFFFF) != 0 )
      {
        v31 = *(_QWORD *)(v30 - 8);
        v32 = 0;
        v33 = (unsigned __int8 **)(v31 + 32LL * *(unsigned int *)(v30 + 72));
        while ( (unsigned __int8 *)v8 != *v33 )
        {
          ++v32;
          ++v33;
          if ( (*(_DWORD *)(v30 + 4) & 0x7FFFFFF) == (_DWORD)v32 )
            goto LABEL_34;
        }
        v34 = *(_QWORD *)(v31 + 32 * v32);
        *v33 = v99;
        v35 = *(_DWORD *)(v30 + 4) & 0x7FFFFFF;
        if ( v35 == *(_DWORD *)(v30 + 72) )
        {
          v103 = v34;
          sub_B48D90(v30);
          v34 = v103;
          v35 = *(_DWORD *)(v30 + 4) & 0x7FFFFFF;
        }
        v36 = (v35 + 1) & 0x7FFFFFF;
        v37 = v36 | *(_DWORD *)(v30 + 4) & 0xF8000000;
        v38 = *(_QWORD *)(v30 - 8) + 32LL * (unsigned int)(v36 - 1);
        *(_DWORD *)(v30 + 4) = v37;
        if ( *(_QWORD *)v38 )
        {
          v39 = *(_QWORD *)(v38 + 8);
          **(_QWORD **)(v38 + 16) = v39;
          if ( v39 )
            *(_QWORD *)(v39 + 16) = *(_QWORD *)(v38 + 16);
        }
        *(_QWORD *)v38 = v34;
        if ( v34 )
        {
          v40 = *(_QWORD *)(v34 + 16);
          *(_QWORD *)(v38 + 8) = v40;
          if ( v40 )
            *(_QWORD *)(v40 + 16) = v38 + 8;
          *(_QWORD *)(v38 + 16) = v34 + 16;
          *(_QWORD *)(v34 + 16) = v38;
        }
        *(_QWORD *)(*(_QWORD *)(v30 - 8)
                  + 32LL * *(unsigned int *)(v30 + 72)
                  + 8LL * ((*(_DWORD *)(v30 + 4) & 0x7FFFFFFu) - 1)) = v100;
      }
LABEL_34:
      v41 = *(_QWORD *)(v30 + 32);
      if ( !v41 )
        BUG();
      v30 = 0;
      if ( *(_BYTE *)(v41 - 24) == 84 )
        v30 = v41 - 24;
    }
    if ( *(_QWORD *)(v4 - 96) )
    {
      v42 = *(_QWORD *)(v4 - 88);
      **(_QWORD **)(v4 - 80) = v42;
      if ( v42 )
        *(_QWORD *)(v42 + 16) = *(_QWORD *)(v4 - 80);
    }
    *(_QWORD *)(v4 - 96) = v8;
    if ( v8 )
    {
      v43 = *(_QWORD *)(v8 + 16);
      v44 = v8 + 16;
      *(_QWORD *)(v4 - 88) = v43;
      if ( v43 )
        *(_QWORD *)(v43 + 16) = v4 - 88;
      *(_QWORD *)(v4 - 80) = v44;
      *(_QWORD *)(v8 + 16) = v4 - 96;
      if ( *(v11 - 12) )
      {
        v45 = *(v11 - 11);
        *(_QWORD *)*(v11 - 10) = v45;
        if ( v45 )
          *(_QWORD *)(v45 + 16) = *(v11 - 10);
      }
      *(v11 - 12) = v8;
      v46 = *(_QWORD *)(v8 + 16);
      *(v11 - 11) = v46;
      if ( v46 )
        *(_QWORD *)(v46 + 16) = v11 - 11;
      *(v11 - 10) = v44;
      *(_QWORD *)(v8 + 16) = v11 - 12;
    }
    else if ( *(v11 - 12) )
    {
      v97 = *(v11 - 11);
      *(_QWORD *)*(v11 - 10) = v97;
      if ( v97 )
        *(_QWORD *)(v97 + 16) = *(v11 - 10);
      *(v11 - 12) = 0;
    }
  }
  v47 = *(_QWORD *)(v4 + 8);
  if ( *(_BYTE *)(v47 + 8) != 7 && *(_QWORD *)(v4 + 16) )
  {
    v48 = *(_QWORD *)(v8 + 56);
    v114 = v8;
    v115 = v48;
    LOWORD(v116) = 1;
    if ( v48 == v8 + 48 )
    {
LABEL_63:
      v109 = 257;
      v124 = 257;
      v56 = sub_BD2DA0(80);
      v57 = v56;
      if ( v56 )
      {
        v58 = v56;
        sub_B44260(v56, v47, 55, 0x8000000u, 0, 0);
        *(_DWORD *)(v57 + 72) = 0;
        sub_BD6B50((unsigned __int8 *)v57, (const char **)&v121);
        sub_BD2A10(v57, *(_DWORD *)(v57 + 72), 1);
      }
      else
      {
        v58 = 0;
      }
      if ( (unsigned __int8)sub_920620(v58) )
      {
        v96 = v119;
        if ( v118 )
          sub_B99FD0(v57, 3u, v118);
        sub_B45150(v57, v96);
      }
      (*(void (__fastcall **)(__int64, __int64, char *, __int64, __int64))(*(_QWORD *)v117 + 16LL))(
        v117,
        v57,
        v108,
        v115,
        v116);
      v61 = v110;
      v62 = &v110[16 * v111];
      if ( v110 != v62 )
      {
        do
        {
          v63 = *((_QWORD *)v61 + 1);
          v64 = *(_DWORD *)v61;
          v61 += 16;
          sub_B99FD0(v57, v64, v63);
        }
        while ( v62 != v61 );
      }
      v65 = *(_QWORD *)(v4 + 16);
      v66 = 0;
      v121 = v123;
      v67 = v65;
      v122 = 0x1000000000LL;
      if ( v65 )
      {
        do
        {
          v67 = *(_QWORD *)(v67 + 8);
          ++v66;
        }
        while ( v67 );
        v68 = v123;
        if ( v66 > 16 )
        {
          sub_C8D5F0((__int64)&v121, v123, v66, 8u, v59, v60);
          v68 = &v121[8 * (unsigned int)v122];
        }
        do
        {
          v68 += 8;
          *((_QWORD *)v68 - 1) = *(_QWORD *)(v65 + 24);
          v65 = *(_QWORD *)(v65 + 8);
        }
        while ( v65 );
        LODWORD(v122) = v122 + v66;
        v69 = &v121[8 * (unsigned int)v122];
        if ( v121 != v69 )
        {
          v70 = (__int64 *)v121;
          do
          {
            v71 = *v70++;
            sub_BD2ED0(v71, v4, v57);
          }
          while ( v69 != (const char *)v70 );
        }
      }
      else
      {
        LODWORD(v122) = 0;
      }
      v72 = *(_QWORD *)(v4 + 40);
      v73 = *(_DWORD *)(v57 + 4) & 0x7FFFFFF;
      if ( v73 == *(_DWORD *)(v57 + 72) )
      {
        sub_B48D90(v57);
        v73 = *(_DWORD *)(v57 + 4) & 0x7FFFFFF;
      }
      v74 = (v73 + 1) & 0x7FFFFFF;
      v75 = v74 | *(_DWORD *)(v57 + 4) & 0xF8000000;
      v76 = *(_QWORD *)(v57 - 8) + 32LL * (unsigned int)(v74 - 1);
      *(_DWORD *)(v57 + 4) = v75;
      if ( *(_QWORD *)v76 )
      {
        v77 = *(_QWORD *)(v76 + 8);
        **(_QWORD **)(v76 + 16) = v77;
        if ( v77 )
          *(_QWORD *)(v77 + 16) = *(_QWORD *)(v76 + 16);
      }
      *(_QWORD *)v76 = v4;
      v78 = *(_QWORD *)(v4 + 16);
      *(_QWORD *)(v76 + 8) = v78;
      if ( v78 )
        *(_QWORD *)(v78 + 16) = v76 + 8;
      *(_QWORD *)(v76 + 16) = v4 + 16;
      *(_QWORD *)(v4 + 16) = v76;
      *(_QWORD *)(*(_QWORD *)(v57 - 8)
                + 32LL * *(unsigned int *)(v57 + 72)
                + 8LL * ((*(_DWORD *)(v57 + 4) & 0x7FFFFFFu) - 1)) = v72;
      v79 = v11[5];
      v80 = *(_DWORD *)(v57 + 4) & 0x7FFFFFF;
      if ( v80 == *(_DWORD *)(v57 + 72) )
      {
        sub_B48D90(v57);
        v80 = *(_DWORD *)(v57 + 4) & 0x7FFFFFF;
      }
      v81 = (v80 + 1) & 0x7FFFFFF;
      v82 = v81 | *(_DWORD *)(v57 + 4) & 0xF8000000;
      v83 = *(_QWORD *)(v57 - 8) + 32LL * (unsigned int)(v81 - 1);
      *(_DWORD *)(v57 + 4) = v82;
      if ( *(_QWORD *)v83 )
      {
        v84 = *(_QWORD *)(v83 + 8);
        **(_QWORD **)(v83 + 16) = v84;
        if ( v84 )
          *(_QWORD *)(v84 + 16) = *(_QWORD *)(v83 + 16);
      }
      *(_QWORD *)v83 = v11;
      v85 = v11[2];
      *(_QWORD *)(v83 + 8) = v85;
      if ( v85 )
        *(_QWORD *)(v85 + 16) = v83 + 8;
      *(_QWORD *)(v83 + 16) = v11 + 2;
      v11[2] = v83;
      *(_QWORD *)(*(_QWORD *)(v57 - 8)
                + 32LL * *(unsigned int *)(v57 + 72)
                + 8LL * ((*(_DWORD *)(v57 + 4) & 0x7FFFFFFu) - 1)) = v79;
      if ( v121 != v123 )
        _libc_free((unsigned __int64)v121);
      goto LABEL_92;
    }
    if ( v48 )
      v48 -= 24;
    v49 = *(const char **)sub_B46C60(v48);
    v121 = (char *)v49;
    if ( v49 && (sub_B96E90((__int64)&v121, (__int64)v49, 1), (v52 = v121) != 0) )
    {
      v53 = v110;
      v54 = v111;
      v55 = &v110[16 * v111];
      if ( v110 != (_BYTE *)v55 )
      {
        while ( *(_DWORD *)v53 )
        {
          v53 += 16;
          if ( v55 == (_QWORD *)v53 )
            goto LABEL_113;
        }
        *((_QWORD *)v53 + 1) = v121;
LABEL_62:
        sub_B91220((__int64)&v121, (__int64)v52);
        v47 = *(_QWORD *)(v4 + 8);
        goto LABEL_63;
      }
LABEL_113:
      if ( v111 >= (unsigned __int64)v112 )
      {
        if ( v112 < (unsigned __int64)v111 + 1 )
        {
          sub_C8D5F0((__int64)&v110, v113, v111 + 1LL, 0x10u, v50, v51);
          v55 = &v110[16 * v111];
        }
        *v55 = 0;
        v55[1] = v52;
        v52 = v121;
        ++v111;
      }
      else
      {
        if ( v55 )
        {
          *(_DWORD *)v55 = 0;
          v55[1] = v52;
          v54 = v111;
          v52 = v121;
        }
        v111 = v54 + 1;
      }
    }
    else
    {
      sub_93FB40((__int64)&v110, 0);
      v52 = v121;
    }
    if ( !v52 )
    {
      v47 = *(_QWORD *)(v4 + 8);
      goto LABEL_63;
    }
    goto LABEL_62;
  }
LABEL_92:
  nullsub_61();
  v120 = &unk_49DA100;
  nullsub_63();
  if ( v110 != v113 )
    _libc_free((unsigned __int64)v110);
  return v11;
}
