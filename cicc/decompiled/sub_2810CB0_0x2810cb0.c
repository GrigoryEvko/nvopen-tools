// Function: sub_2810CB0
// Address: 0x2810cb0
//
__int64 __fastcall sub_2810CB0(__int64 *a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 *a6)
{
  __int64 *v7; // r14
  __int64 v8; // rsi
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 *v11; // r12
  unsigned __int64 *v12; // rbx
  unsigned __int64 *v13; // r12
  unsigned __int64 v14; // rdi
  unsigned __int64 v15; // rbx
  __int64 v16; // r12
  __int64 v17; // rsi
  __int64 v18; // rax
  unsigned int v19; // r8d
  __int64 *v20; // rax
  __int64 v21; // rdx
  __int64 *v22; // r12
  __int64 v23; // r13
  __int64 *v24; // rbx
  __int64 v25; // rax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rcx
  __int64 v29; // rcx
  __int64 v30; // r13
  unsigned __int64 v31; // rax
  int v32; // edx
  _QWORD *v33; // r12
  _QWORD *v34; // rax
  _QWORD *v35; // rax
  _QWORD *v36; // rbx
  __int64 v37; // rsi
  __int64 *v38; // r8
  __int64 v39; // rdx
  unsigned __int64 v40; // rax
  int v41; // edx
  __int64 v42; // r12
  __int64 v43; // rax
  __int64 *v44; // rax
  __int64 v45; // rdx
  __int64 *v46; // r13
  __int64 v47; // r12
  __int64 *v48; // rbx
  __int64 v49; // rbx
  __int64 v50; // rcx
  const char *v51; // r8
  __int64 v52; // rdi
  __int64 v54; // rsi
  __int64 v55; // rax
  unsigned int v56; // r8d
  __int64 *v57; // rax
  __int64 *v58; // r14
  __int64 v59; // r13
  bool v60; // zf
  _QWORD *v61; // r13
  __int64 *v62; // rax
  __int64 v63; // r15
  __int64 v64; // rax
  __int64 v65; // rdx
  __int64 v66; // r13
  __int64 v67; // r15
  const char *v68; // rax
  __int64 v69; // rdx
  __int64 v70; // rbx
  const char *v71; // r15
  __int64 v72; // r13
  __int64 v73; // rdx
  unsigned int v74; // esi
  __int64 v75; // rsi
  unsigned __int8 *v76; // rsi
  __int64 v77; // r11
  __int64 v78; // r10
  unsigned int v79; // ecx
  __int64 v80; // rbx
  const char *v81; // r15
  __int64 v82; // rdx
  unsigned int v83; // esi
  __int64 v84; // rdx
  int v85; // eax
  char v86; // al
  int v87; // edx
  __int64 v88; // rax
  const char *v89; // rax
  __int64 v90; // rdx
  _QWORD *v91; // rdx
  unsigned __int64 v92; // rax
  int v93; // edx
  unsigned __int64 v94; // rax
  __int64 v95; // rdx
  __int64 v96; // rsi
  unsigned int v97; // [rsp+8h] [rbp-2C8h]
  unsigned int v98; // [rsp+Ch] [rbp-2C4h]
  __int64 v100; // [rsp+38h] [rbp-298h]
  __int64 v101; // [rsp+38h] [rbp-298h]
  __int64 v102; // [rsp+40h] [rbp-290h]
  unsigned __int16 v103; // [rsp+48h] [rbp-288h]
  unsigned int v104; // [rsp+48h] [rbp-288h]
  __int64 v105; // [rsp+48h] [rbp-288h]
  __int64 *v106; // [rsp+48h] [rbp-288h]
  __int64 *v107; // [rsp+48h] [rbp-288h]
  __int64 v111; // [rsp+78h] [rbp-258h]
  __int64 *v112; // [rsp+78h] [rbp-258h]
  __int64 *v113; // [rsp+78h] [rbp-258h]
  _QWORD *v114; // [rsp+80h] [rbp-250h] BYREF
  __int64 v115; // [rsp+88h] [rbp-248h]
  __int64 v116[4]; // [rsp+90h] [rbp-240h] BYREF
  __int16 v117; // [rsp+B0h] [rbp-220h]
  __m128i v118; // [rsp+C0h] [rbp-210h] BYREF
  __int64 *v119; // [rsp+D0h] [rbp-200h]
  __int16 v120; // [rsp+E0h] [rbp-1F0h]
  const char *v121; // [rsp+F0h] [rbp-1E0h] BYREF
  __int64 v122; // [rsp+F8h] [rbp-1D8h]
  _BYTE v123[32]; // [rsp+100h] [rbp-1D0h] BYREF
  __int64 v124; // [rsp+120h] [rbp-1B0h]
  __int64 v125; // [rsp+128h] [rbp-1A8h]
  __int64 v126; // [rsp+130h] [rbp-1A0h]
  __int64 v127; // [rsp+138h] [rbp-198h]
  void **v128; // [rsp+140h] [rbp-190h]
  void **v129; // [rsp+148h] [rbp-188h]
  __int64 v130; // [rsp+150h] [rbp-180h] BYREF
  int v131; // [rsp+158h] [rbp-178h]
  __int16 v132; // [rsp+15Ch] [rbp-174h]
  char v133; // [rsp+15Eh] [rbp-172h]
  __int64 v134; // [rsp+160h] [rbp-170h]
  __int64 v135; // [rsp+168h] [rbp-168h]
  void *v136; // [rsp+170h] [rbp-160h] BYREF
  void *v137; // [rsp+178h] [rbp-158h] BYREF

  v7 = a1;
  v8 = a1[1];
  v9 = *(_QWORD *)(**(_QWORD **)(*a1 + 32) + 72LL);
  v10 = **(_QWORD **)(v8 + 32);
  sub_D4BD20(v116, v8, a3, a4, a5, (__int64)a6);
  sub_B157E0((__int64)&v118, v116);
  sub_B17430((__int64)&v121, (__int64)"loop-flatten", (__int64)"Flattened", 9, &v118, v10);
  if ( v116[0] )
    sub_B91220((__int64)v116, v116[0]);
  sub_1049690(v118.m128i_i64, v9);
  sub_B18290((__int64)&v121, "Flattened into outer loop", 0x19u);
  sub_1049740(v118.m128i_i64, (__int64)&v121);
  v11 = v119;
  if ( v119 )
  {
    sub_FDC110(v119);
    j_j___libc_free_0((unsigned __int64)v11);
  }
  v12 = (unsigned __int64 *)v128;
  v121 = (const char *)&unk_49D9D40;
  v13 = (unsigned __int64 *)&v128[10 * (unsigned int)v129];
  if ( v128 != (void **)v13 )
  {
    do
    {
      v13 -= 10;
      v14 = v13[4];
      if ( (unsigned __int64 *)v14 != v13 + 6 )
        j_j___libc_free_0(v14);
      if ( (unsigned __int64 *)*v13 != v13 + 2 )
        j_j___libc_free_0(*v13);
    }
    while ( v12 != v13 );
    v13 = (unsigned __int64 *)v128;
  }
  if ( v13 != (unsigned __int64 *)&v130 )
    _libc_free((unsigned __int64)v13);
  v15 = v7[29];
  if ( !v15 )
  {
    v91 = (_QWORD *)(sub_D4B130(*v7) + 48);
    v92 = *v91 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (_QWORD *)v92 == v91 )
    {
      v15 = 0;
    }
    else
    {
      if ( !v92 )
        BUG();
      v93 = *(unsigned __int8 *)(v92 - 24);
      v94 = v92 - 24;
      if ( (unsigned int)(v93 - 30) < 0xB )
        v15 = v94;
    }
    v95 = v7[5];
    v96 = v7[4];
    v123[17] = 1;
    v121 = "flatten.tripcount";
    v123[16] = 3;
    v7[29] = sub_B504D0(17, v96, v95, (__int64)&v121, v15 + 24, 0);
  }
  v16 = v7[2];
  v17 = sub_D47930(v7[1]);
  if ( (*(_DWORD *)(v16 + 4) & 0x7FFFFFF) != 0 )
  {
    v18 = 0;
    while ( 1 )
    {
      v19 = v18;
      if ( v17 == *(_QWORD *)(*(_QWORD *)(v16 - 8) + 32LL * *(unsigned int *)(v16 + 72) + 8 * v18) )
        break;
      if ( (*(_DWORD *)(v16 + 4) & 0x7FFFFFF) == (_DWORD)++v18 )
        goto LABEL_98;
    }
  }
  else
  {
LABEL_98:
    v19 = -1;
  }
  sub_B48BF0(v16, v19, 1);
  v20 = (__int64 *)v7[19];
  if ( *((_BYTE *)v7 + 172) )
    v21 = *((unsigned int *)v7 + 41);
  else
    v21 = *((unsigned int *)v7 + 40);
  v22 = &v20[v21];
  if ( v20 != v22 )
  {
    while ( 1 )
    {
      v23 = *v20;
      v24 = v20;
      if ( (unsigned __int64)*v20 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v22 == ++v20 )
        goto LABEL_24;
    }
    while ( v22 != v24 )
    {
      v54 = sub_D47930(v7[1]);
      if ( (*(_DWORD *)(v23 + 4) & 0x7FFFFFF) != 0 )
      {
        v55 = 0;
        while ( 1 )
        {
          v56 = v55;
          if ( v54 == *(_QWORD *)(*(_QWORD *)(v23 - 8) + 32LL * *(unsigned int *)(v23 + 72) + 8 * v55) )
            break;
          if ( (*(_DWORD *)(v23 + 4) & 0x7FFFFFF) == (_DWORD)++v55 )
            goto LABEL_72;
        }
      }
      else
      {
LABEL_72:
        v56 = -1;
      }
      sub_B48BF0(v23, v56, 1);
      v57 = v24 + 1;
      if ( v24 + 1 == v22 )
        break;
      while ( 1 )
      {
        v23 = *v57;
        v24 = v57;
        if ( (unsigned __int64)*v57 < 0xFFFFFFFFFFFFFFFELL )
          break;
        if ( v22 == ++v57 )
          goto LABEL_24;
      }
    }
  }
LABEL_24:
  v25 = *(_QWORD *)(v7[17] - 96);
  if ( (*(_BYTE *)(v25 + 7) & 0x40) != 0 )
    v26 = *(_QWORD *)(v25 - 8);
  else
    v26 = v25 - 32LL * (*(_DWORD *)(v25 + 4) & 0x7FFFFFF);
  v27 = v7[29];
  if ( *(_QWORD *)(v26 + 32) )
  {
    v28 = *(_QWORD *)(v26 + 40);
    **(_QWORD **)(v26 + 48) = v28;
    if ( v28 )
      *(_QWORD *)(v28 + 16) = *(_QWORD *)(v26 + 48);
  }
  *(_QWORD *)(v26 + 32) = v27;
  if ( v27 )
  {
    v29 = *(_QWORD *)(v27 + 16);
    *(_QWORD *)(v26 + 40) = v29;
    if ( v29 )
      *(_QWORD *)(v29 + 16) = v26 + 40;
    *(_QWORD *)(v26 + 48) = v27 + 16;
    *(_QWORD *)(v27 + 16) = v26 + 32;
  }
  v111 = sub_D47470(v7[1]);
  v30 = sub_D46F00(v7[1]);
  v31 = *(_QWORD *)(v30 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v31 == v30 + 48 )
  {
    v33 = 0;
  }
  else
  {
    if ( !v31 )
      BUG();
    v32 = *(unsigned __int8 *)(v31 - 24);
    v33 = 0;
    v34 = (_QWORD *)(v31 - 24);
    if ( (unsigned int)(v32 - 30) < 0xB )
      v33 = v34;
  }
  sub_B43C20((__int64)&v121, v30);
  v102 = (__int64)v121;
  v103 = v122;
  v35 = sub_BD2C40(72, 1u);
  v36 = v35;
  if ( v35 )
    sub_B4C8F0((__int64)v35, v111, 1u, v102, v103);
  v37 = v33[6];
  v38 = v36 + 6;
  v121 = (const char *)v37;
  if ( v37 )
  {
    sub_B96E90((__int64)&v121, v37, 1);
    v38 = v36 + 6;
    if ( v36 + 6 == &v121 )
    {
      if ( v121 )
        sub_B91220((__int64)&v121, (__int64)v121);
      goto LABEL_43;
    }
    v75 = v36[6];
    if ( !v75 )
    {
LABEL_104:
      v76 = (unsigned __int8 *)v121;
      v36[6] = v121;
      if ( v76 )
        sub_B976B0((__int64)&v121, v76, (__int64)v38);
      goto LABEL_43;
    }
LABEL_103:
    v113 = v38;
    sub_B91220((__int64)v38, v75);
    v38 = v113;
    goto LABEL_104;
  }
  if ( v38 != (__int64 *)&v121 )
  {
    v75 = v36[6];
    if ( v75 )
      goto LABEL_103;
  }
LABEL_43:
  sub_B43D60(v33);
  sub_B20B50(a2, v30, **(_QWORD **)(v7[1] + 32));
  if ( a6 )
    sub_D6D7F0(a6, v30, **(_QWORD **)(v7[1] + 32));
  v39 = *(_QWORD *)(v7[3] + 40);
  v40 = *(_QWORD *)(v39 + 48) & 0xFFFFFFFFFFFFFFF8LL;
  if ( v40 == v39 + 48 )
  {
    v42 = 0;
  }
  else
  {
    if ( !v40 )
      BUG();
    v41 = *(unsigned __int8 *)(v40 - 24);
    v42 = 0;
    v43 = v40 - 24;
    if ( (unsigned int)(v41 - 30) < 0xB )
      v42 = v43;
  }
  v127 = sub_BD5C60(v42);
  v128 = &v136;
  v129 = &v137;
  v132 = 512;
  v136 = &unk_49DA100;
  v121 = v123;
  v122 = 0x200000000LL;
  v137 = &unk_49DA0B0;
  v130 = 0;
  v131 = 0;
  v133 = 7;
  v134 = 0;
  v135 = 0;
  v124 = 0;
  v125 = 0;
  LOWORD(v126) = 0;
  sub_D5F1F0((__int64)&v121, v42);
  v44 = (__int64 *)v7[7];
  if ( *((_BYTE *)v7 + 76) )
    v45 = *((unsigned int *)v7 + 17);
  else
    v45 = *((unsigned int *)v7 + 16);
  v46 = &v44[v45];
  if ( v44 != v46 )
  {
    while ( 1 )
    {
      v47 = *v44;
      v48 = v44;
      if ( (unsigned __int64)*v44 < 0xFFFFFFFFFFFFFFFELL )
        break;
      if ( v46 == ++v44 )
        goto LABEL_54;
    }
    if ( v46 != v44 )
    {
      v112 = v7;
      v58 = v46;
      while ( 1 )
      {
        v59 = v112[3];
        v60 = *((_BYTE *)v112 + 208) == 0;
        v114 = (_QWORD *)v59;
        if ( v60 )
        {
          if ( *(_BYTE *)v47 != 63 )
            goto LABEL_77;
        }
        else
        {
          v116[0] = (__int64)"flatten.trunciv";
          v117 = 259;
          v63 = *(_QWORD *)(v47 + 8);
          if ( v63 == *(_QWORD *)(v59 + 8) )
          {
            v64 = v59;
          }
          else
          {
            v64 = (*((__int64 (__fastcall **)(void **, __int64, __int64, _QWORD))*v128 + 15))(
                    v128,
                    38,
                    v59,
                    *(_QWORD *)(v47 + 8));
            if ( !v64 )
            {
              v120 = 257;
              v105 = sub_B51D30(38, v59, v63, (__int64)&v118, 0, 0);
              (*((void (__fastcall **)(void **, __int64, __int64 *, __int64, __int64))*v129 + 2))(
                v129,
                v105,
                v116,
                v125,
                v126);
              v64 = v105;
              if ( v121 != &v121[16 * (unsigned int)v122] )
              {
                v106 = v48;
                v70 = (__int64)v121;
                v71 = &v121[16 * (unsigned int)v122];
                v72 = v64;
                do
                {
                  v73 = *(_QWORD *)(v70 + 8);
                  v74 = *(_DWORD *)v70;
                  v70 += 16;
                  sub_B99FD0(v72, v74, v73);
                }
                while ( v71 != (const char *)v70 );
                v48 = v106;
                v64 = v72;
              }
            }
          }
          v114 = (_QWORD *)v64;
          if ( *(_BYTE *)v47 != 63 )
          {
LABEL_77:
            v61 = v114;
            goto LABEL_78;
          }
        }
        v65 = v125;
        v66 = *(_QWORD *)(v47 - 32LL * (*(_DWORD *)(v47 + 4) & 0x7FFFFFF));
        v67 = *(_QWORD *)(v66 - 32LL * (*(_DWORD *)(v66 + 4) & 0x7FFFFFF));
        if ( v125 )
          v65 = v125 - 24;
        if ( (unsigned __int8)sub_B19DB0(a2, *(_QWORD *)(v66 - 32LL * (*(_DWORD *)(v66 + 4) & 0x7FFFFFF)), v65) )
        {
          if ( !sub_B4DE30(v47) )
            goto LABEL_91;
        }
        else
        {
          sub_D5F1F0((__int64)&v121, v47);
          if ( !sub_B4DE30(v47) )
            goto LABEL_91;
        }
        if ( !sub_B4DE30(v66) )
        {
LABEL_91:
          v104 = 0;
          goto LABEL_92;
        }
        v104 = 3;
LABEL_92:
        v68 = sub_BD5D20(v47);
        v117 = 1283;
        v116[0] = (__int64)"flatten.";
        v116[3] = v69;
        v116[2] = (__int64)v68;
        v100 = *(_QWORD *)(v47 + 72);
        v61 = (_QWORD *)(*((__int64 (__fastcall **)(void **, __int64, __int64, _QWORD **, __int64, _QWORD))*v128 + 8))(
                          v128,
                          v100,
                          v67,
                          &v114,
                          1,
                          v104);
        if ( !v61 )
        {
          v120 = 257;
          v61 = sub_BD2C40(88, 2u);
          if ( v61 )
          {
            v77 = *(_QWORD *)(v67 + 8);
            v78 = v100;
            v79 = v98 & 0xE0000000 | 2;
            v98 = v79;
            if ( (unsigned int)*(unsigned __int8 *)(v77 + 8) - 17 <= 1 )
              goto LABEL_115;
            v84 = v114[1];
            v85 = *(unsigned __int8 *)(v84 + 8);
            if ( v85 == 17 )
            {
              v86 = 0;
LABEL_123:
              v87 = *(_DWORD *)(v84 + 32);
              BYTE4(v115) = v86;
              v97 = v79;
              LODWORD(v115) = v87;
              v88 = sub_BCE1B0((__int64 *)v77, v115);
              v79 = v97;
              v78 = v100;
              v77 = v88;
            }
            else if ( v85 == 18 )
            {
              v86 = 1;
              goto LABEL_123;
            }
LABEL_115:
            v101 = v78;
            sub_B44260((__int64)v61, v77, 34, v79, 0, 0);
            v61[9] = v101;
            v61[10] = sub_B4DC50(v101, (__int64)&v114, 1);
            sub_B4D9A0((__int64)v61, v67, (__int64 *)&v114, 1, (__int64)&v118);
          }
          sub_B4DDE0((__int64)v61, v104);
          (*((void (__fastcall **)(void **, _QWORD *, __int64 *, __int64, __int64))*v129 + 2))(
            v129,
            v61,
            v116,
            v125,
            v126);
          if ( v121 != &v121[16 * (unsigned int)v122] )
          {
            v107 = v48;
            v80 = (__int64)v121;
            v81 = &v121[16 * (unsigned int)v122];
            do
            {
              v82 = *(_QWORD *)(v80 + 8);
              v83 = *(_DWORD *)v80;
              v80 += 16;
              sub_B99FD0((__int64)v61, v83, v82);
            }
            while ( v81 != (const char *)v80 );
            v48 = v107;
          }
        }
        v114 = v61;
LABEL_78:
        sub_BD84D0(v47, (__int64)v61);
        v62 = v48 + 1;
        if ( v48 + 1 != v58 )
        {
          while ( 1 )
          {
            v47 = *v62;
            v48 = v62;
            if ( (unsigned __int64)*v62 < 0xFFFFFFFFFFFFFFFELL )
              break;
            if ( v58 == ++v62 )
              goto LABEL_81;
          }
          if ( v58 != v62 )
            continue;
        }
LABEL_81:
        v7 = v112;
        break;
      }
    }
  }
LABEL_54:
  sub_DAC210(a4, *v7);
  sub_D9D700(a4, 0);
  if ( a5 )
  {
    v49 = v7[1];
    v50 = 14;
    v51 = "<unnamed loop>";
    v52 = **(_QWORD **)(v49 + 32);
    if ( v52 && (*(_BYTE *)(v52 + 7) & 0x10) != 0 )
    {
      v89 = sub_BD5D20(v52);
      v49 = v7[1];
      v51 = v89;
      v50 = v90;
    }
    sub_22D0060(*(_QWORD *)(a5 + 8), v49, (__int64)v51, v50);
    if ( v49 == *(_QWORD *)(a5 + 16) )
      *(_BYTE *)(a5 + 24) = 1;
  }
  sub_D4F720(a3, (__int64 *)v7[1]);
  nullsub_61();
  v136 = &unk_49DA100;
  nullsub_63();
  if ( v121 != v123 )
    _libc_free((unsigned __int64)v121);
  return 1;
}
