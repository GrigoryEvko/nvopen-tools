// Function: sub_1172510
// Address: 0x1172510
//
__int64 *__fastcall sub_1172510(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // ecx
  __int64 *v7; // r14
  __int64 v8; // r12
  __int64 *v9; // rax
  __int64 v10; // rbx
  int v11; // edx
  __int64 *v12; // rax
  __int64 v13; // rdx
  __int64 *v14; // r13
  __int64 v15; // r15
  __int64 v16; // rdi
  __int64 *v17; // r13
  __int64 *v18; // rdx
  __int64 *v19; // rsi
  __int64 *v20; // r11
  __int64 v21; // r8
  __int64 v22; // r14
  char v23; // r12
  _BYTE *v24; // r15
  __int64 *v25; // r10
  __int64 *v26; // r15
  unsigned __int8 v28; // al
  int v29; // eax
  __int64 v30; // r9
  unsigned int v31; // edx
  __int64 v32; // r10
  __int64 i; // rax
  __int64 v34; // rdi
  __int64 v35; // rdx
  __int64 v36; // r12
  __int64 v37; // r13
  _QWORD *v38; // rax
  _BYTE *v39; // rdx
  int v40; // r15d
  _QWORD *j; // rdx
  __int64 v42; // r13
  __int64 v43; // r12
  __int64 v44; // r14
  _QWORD *v45; // rdx
  __int64 v46; // rax
  __int64 v47; // rbx
  __int64 v48; // rcx
  int v49; // edx
  int v50; // edx
  unsigned int v51; // esi
  __int64 v52; // rdx
  __int64 v53; // rsi
  __int64 v54; // rsi
  unsigned int v55; // eax
  __int64 v56; // rax
  unsigned int v57; // ecx
  __int64 v58; // rsi
  __int64 *v59; // r13
  __int64 v60; // rbx
  __int64 v61; // rax
  unsigned int v62; // ecx
  __int64 *v63; // rdi
  __int64 *v64; // rdx
  __int64 v65; // rsi
  int v66; // r8d
  __int64 v67; // rax
  __int64 v68; // rcx
  __int64 v69; // rsi
  __int64 v70; // rdx
  const char *v71; // r11
  _QWORD *v72; // r9
  __int64 v73; // r8
  __int64 *v74; // r10
  __int64 v75; // r12
  __int64 v76; // rsi
  __int64 v77; // r15
  __int64 v78; // r14
  int v79; // eax
  unsigned int v80; // edx
  __int64 v81; // rax
  __int64 v82; // rdx
  __int64 v83; // rdx
  __int64 v84; // rbx
  __int64 v85; // r13
  int v86; // eax
  char v87; // dl
  int v88; // esi
  __int64 v89; // [rsp+0h] [rbp-200h]
  __int64 *v90; // [rsp+0h] [rbp-200h]
  __int64 v91; // [rsp+10h] [rbp-1F0h]
  __int64 *v92; // [rsp+10h] [rbp-1F0h]
  __int64 v93; // [rsp+18h] [rbp-1E8h]
  int v94; // [rsp+20h] [rbp-1E0h]
  int v95; // [rsp+20h] [rbp-1E0h]
  _QWORD *v96; // [rsp+20h] [rbp-1E0h]
  unsigned __int8 v97; // [rsp+28h] [rbp-1D8h]
  __int64 *v98; // [rsp+30h] [rbp-1D0h]
  __int64 v99; // [rsp+30h] [rbp-1D0h]
  __int64 v100; // [rsp+30h] [rbp-1D0h]
  const char *v101; // [rsp+30h] [rbp-1D0h]
  __int64 v102; // [rsp+38h] [rbp-1C8h]
  unsigned int v103; // [rsp+38h] [rbp-1C8h]
  _QWORD *v104; // [rsp+38h] [rbp-1C8h]
  unsigned __int8 v105; // [rsp+40h] [rbp-1C0h]
  __int64 v106; // [rsp+40h] [rbp-1C0h]
  const char *v107; // [rsp+40h] [rbp-1C0h]
  __int64 v108; // [rsp+48h] [rbp-1B8h]
  __int64 v109; // [rsp+48h] [rbp-1B8h]
  __int64 v110; // [rsp+48h] [rbp-1B8h]
  char v111; // [rsp+53h] [rbp-1ADh]
  int v112; // [rsp+54h] [rbp-1ACh]
  __int64 v114[4]; // [rsp+60h] [rbp-1A0h] BYREF
  const char *v115; // [rsp+80h] [rbp-180h] BYREF
  _QWORD *v116; // [rsp+88h] [rbp-178h]
  const char *v117; // [rsp+90h] [rbp-170h]
  _QWORD *v118; // [rsp+98h] [rbp-168h]
  __int16 v119; // [rsp+A0h] [rbp-160h]
  __int64 *v120; // [rsp+B0h] [rbp-150h] BYREF
  __int64 v121; // [rsp+B8h] [rbp-148h]
  _BYTE v122[128]; // [rsp+C0h] [rbp-140h] BYREF
  _BYTE *v123; // [rsp+140h] [rbp-C0h] BYREF
  __int64 v124; // [rsp+148h] [rbp-B8h]
  _BYTE v125[176]; // [rsp+150h] [rbp-B0h] BYREF

  v6 = 0;
  v7 = (__int64 *)v122;
  v8 = a2;
  v9 = *(__int64 **)(a2 - 8);
  v10 = *v9;
  v11 = *(_DWORD *)(*v9 + 4);
  v121 = 0x1000000000LL;
  v12 = (__int64 *)v122;
  v120 = (__int64 *)v122;
  v13 = 32LL * (v11 & 0x7FFFFFF);
  v14 = (__int64 *)(v10 - v13);
  v15 = v13 >> 5;
  if ( (unsigned __int64)v13 > 0x200 )
  {
    sub_C8D5F0((__int64)&v120, v122, v13 >> 5, 8u, a5, a6);
    v6 = v121;
    v12 = &v120[(unsigned int)v121];
  }
  if ( (__int64 *)v10 != v14 )
  {
    do
    {
      if ( v12 )
        *v12 = *v14;
      v14 += 4;
      ++v12;
    }
    while ( (__int64 *)v10 != v14 );
    v6 = v121;
  }
  LODWORD(v121) = v6 + v15;
  v108 = a2;
  v112 = sub_B4DE20(v10);
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
  {
    v16 = *(_QWORD *)(a2 - 8);
    a2 = v16 + 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  }
  else
  {
    v16 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  }
  v17 = (__int64 *)sub_116D080(v16, a2, 1);
  v19 = v18;
  if ( v17 == v18 )
    goto LABEL_14;
  v20 = (__int64 *)v122;
  LOBYTE(v21) = 1;
  v22 = v8;
  v23 = 0;
  do
  {
    v24 = (_BYTE *)*v17;
    if ( *(_BYTE *)*v17 != 63
      || (v98 = v20, v105 = v21, v28 = sub_BD36B0(*v17), v20 = v98, !v28)
      || *((_QWORD *)v24 + 9) != *(_QWORD *)(v10 + 72)
      || (*((_DWORD *)v24 + 1) & 0x7FFFFFF) != (*(_DWORD *)(v10 + 4) & 0x7FFFFFF) )
    {
      v7 = v20;
LABEL_14:
      v25 = v120;
      goto LABEL_15;
    }
    v97 = v28;
    v29 = sub_B4DE20((__int64)v24);
    v21 = v105;
    v94 = v29;
    v20 = v98;
    v30 = v97;
    if ( v105 )
    {
      v21 = 0;
      if ( **(_BYTE **)&v24[-32 * (*((_DWORD *)v24 + 1) & 0x7FFFFFF)] == 60 )
      {
        v55 = sub_B4DD90((__int64)v24);
        v30 = v97;
        v20 = v98;
        v21 = v55;
      }
    }
    v31 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
    if ( v31 )
    {
      v32 = v31 - 1;
      for ( i = 0; ; ++i )
      {
        v34 = *(_QWORD *)(v10 + 32 * (i - v31));
        v35 = *(_QWORD *)&v24[32 * (i - (*((_DWORD *)v24 + 1) & 0x7FFFFFF))];
        if ( v35 != v34 )
        {
          if ( *(_BYTE *)v34 == 17 || *(_BYTE *)v35 == 17 || *(_QWORD *)(v35 + 8) != *(_QWORD *)(v34 + 8) || v23 )
          {
            v7 = v20;
            v25 = v120;
LABEL_15:
            v26 = 0;
            goto LABEL_16;
          }
          v120[i] = 0;
          v23 = v30;
        }
        if ( v32 == i )
          break;
        v31 = *(_DWORD *)(v10 + 4) & 0x7FFFFFF;
      }
    }
    v17 += 4;
    v112 &= v94;
  }
  while ( v19 != v17 );
  v36 = v22;
  v111 = v30;
  v7 = v20;
  if ( (_BYTE)v21 )
  {
    v26 = 0;
  }
  else
  {
    v37 = (unsigned int)v121;
    v38 = v125;
    v39 = v125;
    v123 = v125;
    v40 = v121;
    v124 = 0x1000000000LL;
    if ( !(_DWORD)v121 )
      goto LABEL_100;
    if ( (unsigned int)v121 > 0x10uLL )
    {
      sub_C8D5F0((__int64)&v123, v125, (unsigned int)v121, 8u, v21, v30);
      v39 = v123;
      LOBYTE(v21) = 0;
      v38 = &v123[8 * (unsigned int)v124];
    }
    for ( j = &v39[8 * v37]; j != v38; ++v38 )
    {
      if ( v38 )
        *v38 = 0;
    }
    LODWORD(v124) = v40;
    v95 = v121;
    if ( (_DWORD)v121 )
    {
      v102 = v36;
      v42 = 0;
      v43 = v89;
      v91 = v10;
      v90 = v7;
      do
      {
        if ( !v120[v42] )
        {
          v44 = *(_QWORD *)(v91 + 32 * (v42 - (*(_DWORD *)(v91 + 4) & 0x7FFFFFF)));
          v115 = sub_BD5D20(v44);
          v119 = 773;
          v116 = v45;
          v117 = ".pn";
          v99 = *(_QWORD *)(v44 + 8);
          v46 = sub_BD2DA0(80);
          v47 = v46;
          if ( v46 )
          {
            sub_B44260(v46, v99, 55, 0x8000000u, 0, 0);
            *(_DWORD *)(v47 + 72) = v95;
            sub_BD6B50((unsigned __int8 *)v47, &v115);
            sub_BD2A10(v47, *(_DWORD *)(v47 + 72), 1);
          }
          LOWORD(v43) = 0;
          sub_B44220((_QWORD *)v47, v102 + 24, v43);
          v115 = (const char *)v47;
          sub_11715E0(*(_QWORD *)(a1 + 40) + 2096LL, (__int64 *)&v115);
          v48 = *(_QWORD *)(*(_QWORD *)(v102 - 8) + 32LL * *(unsigned int *)(v102 + 72));
          v49 = *(_DWORD *)(v47 + 4) & 0x7FFFFFF;
          if ( v49 == *(_DWORD *)(v47 + 72) )
          {
            v100 = *(_QWORD *)(*(_QWORD *)(v102 - 8) + 32LL * *(unsigned int *)(v102 + 72));
            sub_B48D90(v47);
            v48 = v100;
            v49 = *(_DWORD *)(v47 + 4) & 0x7FFFFFF;
          }
          v50 = (v49 + 1) & 0x7FFFFFF;
          v51 = v50 | *(_DWORD *)(v47 + 4) & 0xF8000000;
          v52 = *(_QWORD *)(v47 - 8) + 32LL * (unsigned int)(v50 - 1);
          *(_DWORD *)(v47 + 4) = v51;
          if ( *(_QWORD *)v52 )
          {
            v53 = *(_QWORD *)(v52 + 8);
            **(_QWORD **)(v52 + 16) = v53;
            if ( v53 )
              *(_QWORD *)(v53 + 16) = *(_QWORD *)(v52 + 16);
          }
          *(_QWORD *)v52 = v44;
          v54 = *(_QWORD *)(v44 + 16);
          *(_QWORD *)(v52 + 8) = v54;
          if ( v54 )
            *(_QWORD *)(v54 + 16) = v52 + 8;
          *(_QWORD *)(v52 + 16) = v44 + 16;
          LOBYTE(v21) = v111;
          *(_QWORD *)(v44 + 16) = v52;
          *(_QWORD *)(*(_QWORD *)(v47 - 8)
                    + 32LL * *(unsigned int *)(v47 + 72)
                    + 8LL * ((*(_DWORD *)(v47 + 4) & 0x7FFFFFFu) - 1)) = v48;
          *(_QWORD *)&v123[8 * v42] = v47;
          v120[v42] = v47;
        }
        ++v42;
      }
      while ( v95 != v42 );
      v10 = v91;
      v36 = v102;
      v7 = v90;
      if ( (_BYTE)v21 )
      {
        v67 = *(_QWORD *)(v102 - 8);
        v68 = *(_DWORD *)(v102 + 4) & 0x7FFFFFF;
        v69 = v102 - 32 * v68;
        if ( (*(_BYTE *)(v102 + 7) & 0x40) != 0 )
        {
          v108 = v67 + 32 * v68;
          v69 = *(_QWORD *)(v102 - 8);
        }
        v70 = *(unsigned int *)(v102 + 72);
        v114[0] = v69;
        v70 *= 32;
        v114[1] = v108;
        v114[2] = v67 + v70;
        v114[3] = v70 + 8 * v68 + v67;
        sub_116D910(&v115, v114, 1);
        v71 = v115;
        v72 = v116;
        v107 = v117;
        v104 = v118;
        if ( v115 != v117 && v116 != v118 )
        {
          v110 = v91;
          v73 = v36;
          v74 = v90;
          do
          {
            v75 = (unsigned int)v124;
            v76 = *v72;
            v77 = 0;
            v78 = *(_QWORD *)v71;
            if ( (_DWORD)v124 )
            {
              do
              {
                v84 = *(_QWORD *)&v123[8 * v77];
                if ( v84 )
                {
                  v85 = *(_QWORD *)(v78 + 32 * (v77 - (*(_DWORD *)(v78 + 4) & 0x7FFFFFF)));
                  v86 = *(_DWORD *)(v84 + 4) & 0x7FFFFFF;
                  if ( v86 == *(_DWORD *)(v84 + 72) )
                  {
                    v92 = v74;
                    v93 = v73;
                    v96 = v72;
                    v101 = v71;
                    sub_B48D90(*(_QWORD *)&v123[8 * v77]);
                    v74 = v92;
                    v73 = v93;
                    v72 = v96;
                    v71 = v101;
                    v86 = *(_DWORD *)(v84 + 4) & 0x7FFFFFF;
                  }
                  v79 = (v86 + 1) & 0x7FFFFFF;
                  v80 = v79 | *(_DWORD *)(v84 + 4) & 0xF8000000;
                  v81 = *(_QWORD *)(v84 - 8) + 32LL * (unsigned int)(v79 - 1);
                  *(_DWORD *)(v84 + 4) = v80;
                  if ( *(_QWORD *)v81 )
                  {
                    v82 = *(_QWORD *)(v81 + 8);
                    **(_QWORD **)(v81 + 16) = v82;
                    if ( v82 )
                      *(_QWORD *)(v82 + 16) = *(_QWORD *)(v81 + 16);
                  }
                  *(_QWORD *)v81 = v85;
                  if ( v85 )
                  {
                    v83 = *(_QWORD *)(v85 + 16);
                    *(_QWORD *)(v81 + 8) = v83;
                    if ( v83 )
                      *(_QWORD *)(v83 + 16) = v81 + 8;
                    *(_QWORD *)(v81 + 16) = v85 + 16;
                    *(_QWORD *)(v85 + 16) = v81;
                  }
                  *(_QWORD *)(*(_QWORD *)(v84 - 8)
                            + 32LL * *(unsigned int *)(v84 + 72)
                            + 8LL * ((*(_DWORD *)(v84 + 4) & 0x7FFFFFFu) - 1)) = v76;
                }
                ++v77;
              }
              while ( v75 != v77 );
            }
            ++v72;
            v71 += 32;
          }
          while ( v72 != v104 && v107 != v71 );
          v10 = v110;
          v36 = v73;
          v7 = v74;
        }
      }
      v56 = (unsigned int)v121;
      v57 = v121;
    }
    else
    {
LABEL_100:
      v56 = 0;
      v57 = 0;
    }
    v109 = v56 - 1;
    v58 = *v120;
    v103 = v57;
    v59 = v120 + 1;
    v119 = 257;
    v60 = *(_QWORD *)(v10 + 72);
    v106 = v58;
    v26 = sub_BD2C40(88, v57);
    if ( v26 )
    {
      v61 = *(_QWORD *)(v58 + 8);
      v62 = v103 & 0x7FFFFFF;
      if ( (unsigned int)*(unsigned __int8 *)(v61 + 8) - 17 > 1 )
      {
        v63 = &v59[v109];
        if ( v59 != v63 )
        {
          v64 = v59;
          while ( 1 )
          {
            v65 = *(_QWORD *)(*v64 + 8);
            v66 = *(unsigned __int8 *)(v65 + 8);
            if ( v66 == 17 )
            {
              v87 = 0;
              goto LABEL_98;
            }
            if ( v66 == 18 )
              break;
            if ( v63 == ++v64 )
              goto LABEL_64;
          }
          v87 = 1;
LABEL_98:
          v88 = *(_DWORD *)(v65 + 32);
          BYTE4(v114[0]) = v87;
          LODWORD(v114[0]) = v88;
          v61 = sub_BCE1B0((__int64 *)v61, v114[0]);
          v62 = v103 & 0x7FFFFFF;
        }
      }
LABEL_64:
      sub_B44260((__int64)v26, v61, 34, v62, 0, 0);
      v26[9] = v60;
      v26[10] = sub_B4DC50(v60, (__int64)v59, v109);
      sub_B4D9A0((__int64)v26, v106, v59, v109, (__int64)&v115);
    }
    sub_B4DDE0((__int64)v26, v112);
    v19 = v26;
    sub_116D800(a1, (__int64)v26, v36);
    if ( v123 != v125 )
    {
      _libc_free(v123, v26);
      v25 = v120;
      goto LABEL_16;
    }
  }
  v25 = v120;
LABEL_16:
  if ( v25 != v7 )
    _libc_free(v25, v19);
  return v26;
}
