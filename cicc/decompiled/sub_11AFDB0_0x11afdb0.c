// Function: sub_11AFDB0
// Address: 0x11afdb0
//
_QWORD *__fastcall sub_11AFDB0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v3; // rbx
  unsigned __int8 *v4; // r14
  int v5; // eax
  _QWORD *result; // rax
  int v7; // eax
  __int64 *v8; // rdx
  __int64 v9; // r13
  __int64 v10; // r15
  __int64 v11; // rax
  __int64 v12; // r15
  __int64 v13; // rdx
  __int64 v14; // r8
  unsigned __int8 v15; // dl
  unsigned int v16; // ecx
  __int64 *v17; // rdx
  __int64 v18; // r9
  __int64 v19; // r10
  __int64 v20; // rdx
  __int64 v21; // rsi
  unsigned __int64 v22; // rcx
  __int64 v23; // rax
  __int64 v24; // rdx
  __int64 v25; // rax
  unsigned int **v26; // rdi
  __int64 v27; // rax
  __int64 v28; // r13
  __int64 v29; // r12
  unsigned int v30; // ecx
  __int64 v31; // rax
  __int64 v32; // r12
  unsigned __int8 v33; // al
  _QWORD *v34; // rax
  __int64 v35; // rax
  __int64 *v36; // rbx
  __int64 v37; // r13
  __int64 v38; // rbx
  __int64 v39; // rdx
  unsigned int v40; // esi
  int v41; // eax
  unsigned __int8 v42; // dl
  int v43; // ecx
  unsigned __int8 v44; // al
  char v45; // dl
  unsigned int v46; // esi
  __int64 v47; // rax
  char v48; // dl
  __int64 v49; // r10
  __int64 v50; // rax
  unsigned int **v51; // rdi
  __int64 v52; // r11
  _QWORD *v53; // rax
  __int64 v54; // rax
  __int64 *v55; // rbx
  __int64 v56; // r13
  __int64 v57; // r13
  __int64 v58; // rbx
  __int64 v59; // rdx
  unsigned int v60; // esi
  __int64 v61; // rax
  __int64 v62; // rdx
  unsigned __int64 v63; // rax
  bool v64; // al
  unsigned int v65; // eax
  __int64 v66; // rax
  __int64 v67; // rax
  _QWORD *v68; // rax
  __int64 v69; // rax
  __int64 *v70; // r13
  __int64 v71; // r10
  __int64 v72; // rbx
  __int64 v73; // rsi
  char v74; // al
  __int64 v75; // r11
  __int64 v76; // rdx
  int v77; // r8d
  __int64 v78; // rax
  __int64 v79; // r13
  __int64 v80; // r12
  __int64 v81; // rbx
  __int64 v82; // rdx
  unsigned int v83; // esi
  __int64 v84; // [rsp+0h] [rbp-E0h]
  unsigned int v85; // [rsp+Ch] [rbp-D4h]
  __int64 v86; // [rsp+10h] [rbp-D0h]
  __int64 v87; // [rsp+18h] [rbp-C8h]
  __int64 v88; // [rsp+20h] [rbp-C0h]
  unsigned __int64 v89; // [rsp+28h] [rbp-B8h]
  __int64 v90; // [rsp+28h] [rbp-B8h]
  __int64 v91; // [rsp+28h] [rbp-B8h]
  int v92; // [rsp+28h] [rbp-B8h]
  __int64 v93; // [rsp+28h] [rbp-B8h]
  char v94; // [rsp+30h] [rbp-B0h]
  int v95; // [rsp+30h] [rbp-B0h]
  char v96; // [rsp+30h] [rbp-B0h]
  __int64 v97; // [rsp+30h] [rbp-B0h]
  __int64 v98; // [rsp+30h] [rbp-B0h]
  __int64 v99; // [rsp+30h] [rbp-B0h]
  __int64 v100; // [rsp+30h] [rbp-B0h]
  __int64 v101; // [rsp+30h] [rbp-B0h]
  unsigned int v102; // [rsp+38h] [rbp-A8h]
  char v103; // [rsp+40h] [rbp-A0h]
  int v104; // [rsp+40h] [rbp-A0h]
  unsigned int v105; // [rsp+40h] [rbp-A0h]
  __int64 v106; // [rsp+40h] [rbp-A0h]
  int v107; // [rsp+48h] [rbp-98h]
  unsigned int v108; // [rsp+48h] [rbp-98h]
  _QWORD *v109; // [rsp+48h] [rbp-98h]
  _QWORD *v110; // [rsp+48h] [rbp-98h]
  __int64 v111; // [rsp+48h] [rbp-98h]
  char v112; // [rsp+48h] [rbp-98h]
  __int64 v113; // [rsp+48h] [rbp-98h]
  _QWORD *v114; // [rsp+48h] [rbp-98h]
  int v115; // [rsp+48h] [rbp-98h]
  __int64 v116; // [rsp+48h] [rbp-98h]
  _QWORD *v117; // [rsp+48h] [rbp-98h]
  __int64 v118; // [rsp+48h] [rbp-98h]
  __int64 v119; // [rsp+48h] [rbp-98h]
  int v120; // [rsp+48h] [rbp-98h]
  _QWORD v121[4]; // [rsp+50h] [rbp-90h] BYREF
  __int16 v122; // [rsp+70h] [rbp-70h]
  __int64 v123; // [rsp+80h] [rbp-60h] BYREF
  __int64 v124; // [rsp+88h] [rbp-58h]
  __int16 v125; // [rsp+A0h] [rbp-40h]

  v2 = a1;
  v3 = a2;
  v4 = *(unsigned __int8 **)(a2 - 64);
  v5 = *v4;
  if ( (unsigned __int8)v5 > 0x1Cu )
  {
    v7 = v5 - 29;
  }
  else
  {
    if ( (_BYTE)v5 != 5 )
      return 0;
    v7 = *((unsigned __int16 *)v4 + 1);
  }
  if ( v7 != 49 )
    return 0;
  v8 = (v4[7] & 0x40) != 0 ? (__int64 *)*((_QWORD *)v4 - 1) : (__int64 *)&v4[-32 * (*((_DWORD *)v4 + 1) & 0x7FFFFFF)];
  v9 = *v8;
  if ( !*v8 )
    return 0;
  v10 = *(_QWORD *)(a2 - 32);
  if ( *(_BYTE *)v10 != 17 )
    return 0;
  if ( *(_DWORD *)(v10 + 32) > 0x40u )
  {
    v107 = *(_DWORD *)(v10 + 32);
    if ( v107 - (unsigned int)sub_C444A0(v10 + 24) > 0x40 )
      return 0;
    v89 = **(_QWORD **)(v10 + 24);
  }
  else
  {
    v89 = *(_QWORD *)(v10 + 24);
  }
  v11 = *((_QWORD *)v4 + 1);
  v12 = *(_QWORD *)(a2 + 8);
  v94 = *(_BYTE *)(v11 + 8);
  v108 = *(_DWORD *)(v11 + 32);
  v123 = sub_BCAE30(v12);
  v124 = v13;
  v102 = sub_CA1930(&v123);
  v103 = **(_BYTE **)(a1 + 88);
  if ( !(unsigned __int8)sub_F0C3D0(a1) )
  {
    v14 = *(_QWORD *)(v9 + 8);
    v15 = *(_BYTE *)(v14 + 8);
    if ( v15 != 12 )
      goto LABEL_15;
    if ( v103 )
      v89 = v108 - 1 - v89;
    v30 = v89 * v102;
    if ( !((_DWORD)v89 * v102)
      || (v61 = sub_BCAE30(v14),
          v124 = v62,
          v123 = v61,
          v63 = sub_CA1930(&v123),
          v64 = sub_F0C740(a1, v63),
          v30 = v89 * v102,
          v64) )
    {
      v31 = *(_QWORD *)(*(_QWORD *)(a2 - 64) + 16LL);
      if ( v31 )
      {
        if ( !*(_QWORD *)(v31 + 8) )
        {
          if ( v30 )
          {
            v121[0] = "extelt.offset";
            v72 = *(_QWORD *)(a1 + 32);
            v122 = 259;
            v118 = sub_AD64C0(*(_QWORD *)(v9 + 8), v30, 0);
            v73 = (*(__int64 (__fastcall **)(_QWORD, __int64, __int64, __int64, _QWORD))(**(_QWORD **)(v72 + 80) + 24LL))(
                    *(_QWORD *)(v72 + 80),
                    26,
                    v9,
                    v118,
                    0);
            if ( !v73 )
            {
              v125 = 257;
              v119 = sub_B504D0(26, v9, v118, (__int64)&v123, 0, 0);
              (*(void (__fastcall **)(_QWORD, __int64, _QWORD *, _QWORD, _QWORD))(**(_QWORD **)(v72 + 88) + 16LL))(
                *(_QWORD *)(v72 + 88),
                v119,
                v121,
                *(_QWORD *)(v72 + 56),
                *(_QWORD *)(v72 + 64));
              sub_94AAF0((unsigned int **)v72, v119);
              v73 = v119;
            }
            v9 = v73;
          }
          v33 = *(_BYTE *)(v12 + 8);
          if ( v33 > 3u && v33 != 5 && (v33 & 0xFD) != 4 )
          {
            v125 = 257;
            result = sub_BD2C40(72, unk_3F10A14);
            if ( result )
            {
              v114 = result;
              sub_B51510((__int64)result, v9, v12, (__int64)&v123, 0, 0);
              return v114;
            }
            return result;
          }
          v34 = (_QWORD *)sub_BD5C60(v9);
          v35 = sub_BCD140(v34, v102);
          v36 = *(__int64 **)(a1 + 32);
          v122 = 257;
          if ( v35 == *(_QWORD *)(v9 + 8) )
          {
            v32 = v9;
          }
          else
          {
            v111 = v35;
            v32 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v36[10] + 120LL))(
                    v36[10],
                    38,
                    v9,
                    v35);
            if ( !v32 )
            {
              v125 = 257;
              v32 = sub_B51D30(38, v9, v111, (__int64)&v123, 0, 0);
              (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v36[11] + 16LL))(
                v36[11],
                v32,
                v121,
                v36[7],
                v36[8]);
              v37 = *v36 + 16LL * *((unsigned int *)v36 + 2);
              if ( *v36 != v37 )
              {
                v38 = *v36;
                do
                {
                  v39 = *(_QWORD *)(v38 + 8);
                  v40 = *(_DWORD *)v38;
                  v38 += 16;
                  sub_B99FD0(v32, v40, v39);
                }
                while ( v37 != v38 );
              }
            }
          }
LABEL_43:
          v125 = 257;
          result = sub_BD2C40(72, unk_3F10A14);
          if ( result )
          {
            v110 = result;
            sub_B51BF0((__int64)result, v32, v12, (__int64)&v123, 0, 0);
            return v110;
          }
          return result;
        }
      }
    }
  }
  v14 = *(_QWORD *)(v9 + 8);
  v15 = *(_BYTE *)(v14 + 8);
LABEL_15:
  if ( (unsigned int)v15 - 17 > 1 )
    return 0;
  v16 = *(_DWORD *)(v14 + 32);
  if ( v108 == v16 )
  {
    if ( (v94 == 18) != (v15 == 18) )
      return 0;
    v32 = sub_9B7C00(v9, (unsigned int)v89);
    if ( !v32 )
      return 0;
    goto LABEL_43;
  }
  if ( v108 <= v16 || *(_BYTE *)v9 != 91 )
    return 0;
  if ( (*(_BYTE *)(v9 + 7) & 0x40) != 0 )
  {
    v17 = *(__int64 **)(v9 - 8);
    v18 = *v17;
    if ( !*v17 )
      return 0;
  }
  else
  {
    v17 = (__int64 *)(v9 - 32LL * (*(_DWORD *)(v9 + 4) & 0x7FFFFFF));
    v18 = *v17;
    if ( !*v17 )
      return 0;
  }
  v19 = v17[4];
  if ( !v19 )
    return 0;
  v20 = v17[8];
  if ( *(_BYTE *)v20 != 17 )
    return 0;
  if ( *(_DWORD *)(v20 + 32) > 0x40u )
  {
    v84 = v19;
    v85 = *(_DWORD *)(v14 + 32);
    v86 = v14;
    v87 = v18;
    v88 = v20;
    v95 = *(_DWORD *)(v20 + 32);
    if ( v95 - (unsigned int)sub_C444A0(v20 + 24) > 0x40 )
      return 0;
    v18 = v87;
    v14 = v86;
    v16 = v85;
    v19 = v84;
    v21 = **(_QWORD **)(v88 + 24);
  }
  else
  {
    v21 = *(_QWORD *)(v20 + 24);
  }
  v22 = v108 / v16;
  if ( v89 / v22 == v21 )
  {
    v41 = v22 - 1 - v89 % v22;
    if ( !v103 )
      v41 = v89 % v22;
    v42 = *(_BYTE *)(v12 + 8);
    v43 = v41;
    v44 = *(_BYTE *)(**(_QWORD **)(v14 + 16) + 8LL);
    if ( v44 <= 3u || v44 == 5 )
    {
      if ( v42 <= 3u || v42 == 5 )
        return 0;
    }
    else
    {
      if ( (v44 & 0xFD) != 4 )
      {
        if ( v42 > 3u && v42 != 5 )
        {
          if ( (v42 & 0xFD) == 4 )
          {
            v106 = v19;
            v115 = v43;
            v65 = sub_BCB060(v14);
            v49 = v106;
            v46 = v65;
            v66 = *(_QWORD *)(v9 + 16);
            if ( !v66 )
              return 0;
            v105 = v102 * v115;
            if ( *(_QWORD *)(v66 + 8) )
              return 0;
            v67 = *(_QWORD *)(*(_QWORD *)(v3 - 64) + 16LL);
            if ( !v67 )
              return 0;
            v112 = 1;
            v48 = 0;
            goto LABEL_96;
          }
          v101 = v19;
          v120 = v43;
          v46 = sub_BCB060(v14);
          v47 = *(_QWORD *)(v9 + 16);
          v49 = v101;
          v105 = v102 * v120;
          if ( !v47 )
            goto LABEL_65;
          v112 = 0;
          v48 = 0;
LABEL_62:
          if ( *(_QWORD *)(v47 + 8) || (v67 = *(_QWORD *)(*(_QWORD *)(v3 - 64) + 16LL)) == 0 )
          {
LABEL_63:
            if ( v112 || v48 )
              return 0;
LABEL_65:
            v112 = 0;
            if ( !v105 )
              goto LABEL_106;
            goto LABEL_66;
          }
LABEL_96:
          if ( !*(_QWORD *)(v67 + 8) )
          {
            if ( !v48 )
            {
              v52 = v49;
              if ( !v105 )
                goto LABEL_69;
              goto LABEL_68;
            }
            v97 = v49;
            v68 = (_QWORD *)sub_BD5C60(v49);
            v69 = sub_BCD140(v68, v46);
            v70 = *(__int64 **)(a1 + 32);
            v91 = v69;
            v122 = 257;
            if ( v69 == *(_QWORD *)(v97 + 8) )
            {
              v52 = v97;
            }
            else
            {
              v52 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v70[10] + 120LL))(
                      v70[10],
                      49,
                      v97,
                      v69);
              if ( !v52 )
              {
                v125 = 257;
                v98 = sub_B51D30(49, v97, v91, (__int64)&v123, 0, 0);
                v74 = sub_920620(v98);
                v75 = v98;
                if ( v74 )
                {
                  v76 = v70[12];
                  v77 = *((_DWORD *)v70 + 26);
                  if ( v76 )
                  {
                    v92 = *((_DWORD *)v70 + 26);
                    sub_B99FD0(v98, 3u, v76);
                    v77 = v92;
                    v75 = v98;
                  }
                  v99 = v75;
                  sub_B45150(v75, v77);
                  v75 = v99;
                }
                v100 = v75;
                (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v70[11] + 16LL))(
                  v70[11],
                  v75,
                  v121,
                  v70[7],
                  v70[8]);
                v78 = *v70;
                v52 = v100;
                v79 = 16LL * *((unsigned int *)v70 + 2);
                if ( v78 != v78 + v79 )
                {
                  v80 = v78;
                  v93 = v3;
                  v81 = v78 + v79;
                  do
                  {
                    v82 = *(_QWORD *)(v80 + 8);
                    v83 = *(_DWORD *)v80;
                    v80 += 16;
                    sub_B99FD0(v100, v83, v82);
                  }
                  while ( v81 != v80 );
                  v2 = a1;
                  v3 = v93;
                  v52 = v100;
                }
              }
            }
            if ( !v105 )
              goto LABEL_69;
            v49 = v52;
LABEL_66:
            v50 = *(_QWORD *)(*(_QWORD *)(v3 - 64) + 16LL);
            if ( !v50 || *(_QWORD *)(v50 + 8) )
              return 0;
LABEL_68:
            v51 = *(unsigned int ***)(v2 + 32);
            v125 = 257;
            v52 = sub_920DA0(v51, v49, v105, (__int64)&v123, 0);
LABEL_69:
            if ( v112 )
            {
              v113 = v52;
              v53 = (_QWORD *)sub_BD5C60(v52);
              v54 = sub_BCD140(v53, v102);
              v55 = *(__int64 **)(v2 + 32);
              v122 = 257;
              v56 = v54;
              if ( v54 == *(_QWORD *)(v113 + 8) )
              {
                v32 = v113;
              }
              else
              {
                v32 = (*(__int64 (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)v55[10] + 120LL))(
                        v55[10],
                        38,
                        v113,
                        v54);
                if ( !v32 )
                {
                  v125 = 257;
                  v32 = sub_B51D30(38, v113, v56, (__int64)&v123, 0, 0);
                  (*(void (__fastcall **)(__int64, __int64, _QWORD *, __int64, __int64))(*(_QWORD *)v55[11] + 16LL))(
                    v55[11],
                    v32,
                    v121,
                    v55[7],
                    v55[8]);
                  v57 = *v55 + 16LL * *((unsigned int *)v55 + 2);
                  if ( *v55 != v57 )
                  {
                    v58 = *v55;
                    do
                    {
                      v59 = *(_QWORD *)(v58 + 8);
                      v60 = *(_DWORD *)v58;
                      v58 += 16;
                      sub_B99FD0(v32, v60, v59);
                    }
                    while ( v57 != v58 );
                  }
                }
              }
              goto LABEL_43;
            }
            v49 = v52;
LABEL_106:
            v116 = v49;
            v125 = 257;
            result = sub_BD2C40(72, unk_3F10A14);
            if ( result )
            {
              v71 = v116;
              v117 = result;
              sub_B51510((__int64)result, v71, v12, (__int64)&v123, 0, 0);
              return v117;
            }
            return result;
          }
          goto LABEL_63;
        }
        v112 = 1;
        v45 = 0;
LABEL_61:
        v90 = v19;
        v104 = v43;
        v96 = v45;
        v46 = sub_BCB060(v14);
        v47 = *(_QWORD *)(v9 + 16);
        v48 = v96;
        v49 = v90;
        v105 = v102 * v104;
        if ( !v47 )
          return 0;
        goto LABEL_62;
      }
      result = 0;
      if ( v42 <= 3u || v42 == 5 )
        return result;
    }
    if ( (v42 & 0xFD) == 4 )
      return 0;
    v112 = 0;
    v45 = 1;
    goto LABEL_61;
  }
  v23 = *(_QWORD *)(v9 + 16);
  if ( !v23 )
    return 0;
  if ( *(_QWORD *)(v23 + 8) )
    return 0;
  v24 = *(_QWORD *)(v3 - 64);
  v25 = *(_QWORD *)(v24 + 16);
  if ( !v25 || *(_QWORD *)(v25 + 8) )
    return 0;
  v26 = *(unsigned int ***)(a1 + 32);
  v125 = 257;
  v27 = sub_A83570(v26, v18, *(_QWORD *)(v24 + 8), (__int64)&v123);
  v125 = 257;
  v28 = *(_QWORD *)(v3 - 32);
  v29 = v27;
  result = sub_BD2C40(72, 2u);
  if ( result )
  {
    v109 = result;
    sub_B4DE80((__int64)result, v29, v28, (__int64)&v123, 0, 0);
    return v109;
  }
  return result;
}
