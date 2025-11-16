// Function: sub_39207C0
// Address: 0x39207c0
//
__int64 __fastcall sub_39207C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v5; // r14
  __int64 v7; // rbx
  __int64 v8; // r12
  int v9; // eax
  __int64 v10; // r13
  int v11; // edx
  __int64 v12; // r13
  __int64 v13; // rax
  unsigned int v14; // esi
  __int64 v15; // r9
  unsigned int v16; // r8d
  __int64 *v17; // rax
  __int64 v18; // rdi
  unsigned int v19; // r8d
  _BYTE *v20; // rsi
  unsigned int v21; // r10d
  __int64 v22; // r9
  unsigned int v23; // ecx
  __int64 *v24; // rdx
  _BYTE *v25; // rdi
  __int64 v26; // r8
  char v27; // cl
  unsigned __int64 v28; // rax
  _BYTE *v29; // rsi
  unsigned int v30; // edx
  _BYTE *v31; // rdi
  int v32; // edx
  __int64 v33; // r13
  unsigned __int64 v34; // rdx
  int v35; // edi
  _BYTE *v36; // rsi
  char v37; // cl
  __int64 v38; // rax
  int v39; // edx
  int v40; // r8d
  _BYTE *v41; // rax
  __int64 v42; // rax
  unsigned __int64 v43; // rax
  unsigned int v44; // esi
  __int64 v45; // r9
  unsigned int v46; // r8d
  __int64 *v47; // rcx
  __int64 v48; // rdi
  int v49; // edx
  unsigned __int64 v50; // rax
  _BYTE *v51; // rsi
  __int64 v52; // rsi
  unsigned int v53; // ecx
  __int64 v54; // r8
  _BYTE *v55; // r11
  int i; // r13d
  __int64 *v57; // rax
  __int64 v58; // rdx
  _QWORD *v59; // rax
  __int64 *v60; // rcx
  int v61; // eax
  int v62; // eax
  int v63; // r11d
  int v64; // r11d
  __int64 v65; // r10
  unsigned int v66; // eax
  int v67; // ecx
  __int64 *v68; // rdx
  __int64 v69; // r8
  int v70; // r13d
  __int64 *v71; // r11
  int v72; // eax
  __int64 v73; // rdi
  int v74; // eax
  __int64 v75; // rax
  int v76; // r10d
  int v77; // r10d
  __int64 v78; // r8
  unsigned int v79; // edx
  __int64 v80; // rsi
  int v81; // r11d
  __int64 *v82; // rdi
  int v83; // edi
  int v84; // r11d
  int v85; // r11d
  __int64 v86; // r10
  int v87; // edi
  unsigned int v88; // eax
  __int64 *v89; // rsi
  __int64 v90; // r8
  int v91; // r10d
  int v92; // r10d
  __int64 v93; // r8
  int v94; // r11d
  unsigned int v95; // edx
  __int64 v96; // rsi
  int v97; // r13d
  __int64 v98; // r8
  int v99; // r13d
  __int64 v100; // r10
  unsigned int v101; // edx
  __int64 v102; // rdi
  int v103; // esi
  __int64 *v104; // rcx
  int v105; // r13d
  __int64 v106; // r8
  int v107; // r13d
  int v108; // esi
  __int64 v109; // r10
  unsigned int v110; // edx
  __int64 v111; // rdi
  int v112; // r11d
  __int64 v113; // r13
  int v114; // edi
  __int64 *v115; // [rsp+0h] [rbp-80h]
  __int64 *v116; // [rsp+0h] [rbp-80h]
  unsigned int v117; // [rsp+8h] [rbp-78h]
  int v118; // [rsp+8h] [rbp-78h]
  int v119; // [rsp+8h] [rbp-78h]
  unsigned int v120; // [rsp+8h] [rbp-78h]
  unsigned int v121; // [rsp+8h] [rbp-78h]
  __int64 v122; // [rsp+10h] [rbp-70h]
  _QWORD v124[2]; // [rsp+20h] [rbp-60h] BYREF
  _QWORD v125[2]; // [rsp+30h] [rbp-50h] BYREF
  __int16 v126; // [rsp+40h] [rbp-40h]

  result = a2 + 40 * a3;
  v5 = *(_QWORD *)(a1 + 8);
  v122 = result;
  if ( a2 != result )
  {
    v7 = a2;
    while ( 2 )
    {
      v8 = *(_QWORD *)(*(_QWORD *)(v7 + 32) + 184LL) + *(_QWORD *)v7 + a4;
      v9 = *(_DWORD *)(v7 + 24);
      switch ( v9 )
      {
        case 0:
        case 7:
          v19 = *(_DWORD *)(a1 + 184);
          v20 = *(_BYTE **)(v7 + 8);
          if ( !v19 )
            goto LABEL_68;
          v21 = v19 - 1;
          v22 = *(_QWORD *)(a1 + 168);
          v23 = (v19 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v24 = (__int64 *)(v22 + 16LL * v23);
          v25 = (_BYTE *)*v24;
          if ( v20 == (_BYTE *)*v24 )
          {
            v11 = *((_DWORD *)v24 + 2);
            goto LABEL_16;
          }
          v117 = (v19 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
          v55 = (_BYTE *)*v24;
          for ( i = 1; ; ++i )
          {
            if ( v55 == (_BYTE *)-8LL )
            {
LABEL_68:
              if ( (*v20 & 4) != 0 )
              {
                v57 = (__int64 *)*((_QWORD *)v20 - 1);
                v58 = *v57;
                v59 = v57 + 2;
              }
              else
              {
                v58 = 0;
                v59 = 0;
              }
              v124[0] = v59;
              v126 = 1283;
              v125[0] = "symbol not found in wasm index space: ";
              v124[1] = v58;
              v125[1] = v124;
              sub_16BCFB0((__int64)v125, 1u);
            }
            v117 = v21 & (v117 + i);
            v55 = *(_BYTE **)(v22 + 16LL * v117);
            if ( v20 == v55 )
              break;
          }
          v70 = 1;
          v71 = 0;
          while ( 2 )
          {
            if ( v25 == (_BYTE *)-8LL )
            {
              v72 = *(_DWORD *)(a1 + 176);
              v73 = a1 + 160;
              if ( !v71 )
                v71 = v24;
              ++*(_QWORD *)(a1 + 160);
              v74 = v72 + 1;
              if ( 4 * v74 >= 3 * v19 )
              {
                sub_391E830(v73, 2 * v19);
                v97 = *(_DWORD *)(a1 + 184);
                if ( v97 )
                {
                  v98 = *(_QWORD *)(v7 + 8);
                  v99 = v97 - 1;
                  v100 = *(_QWORD *)(a1 + 168);
                  v101 = v99 & (((unsigned int)v98 >> 9) ^ ((unsigned int)v98 >> 4));
                  v74 = *(_DWORD *)(a1 + 176) + 1;
                  v71 = (__int64 *)(v100 + 16LL * v101);
                  v102 = *v71;
                  if ( *v71 == v98 )
                    goto LABEL_94;
                  v103 = 1;
                  v104 = 0;
                  while ( v102 != -8 )
                  {
                    if ( !v104 && v102 == -16 )
                      v104 = v71;
                    v101 = v99 & (v103 + v101);
                    v71 = (__int64 *)(v100 + 16LL * v101);
                    v102 = *v71;
                    if ( v98 == *v71 )
                      goto LABEL_94;
                    ++v103;
                  }
LABEL_130:
                  if ( v104 )
                    v71 = v104;
                  goto LABEL_94;
                }
              }
              else
              {
                if ( v19 - *(_DWORD *)(a1 + 180) - v74 > v19 >> 3 )
                {
LABEL_94:
                  *(_DWORD *)(a1 + 176) = v74;
                  if ( *v71 != -8 )
                    --*(_DWORD *)(a1 + 180);
                  v75 = *(_QWORD *)(v7 + 8);
                  v11 = 0;
                  *((_DWORD *)v71 + 2) = 0;
                  *v71 = v75;
                  v9 = *(_DWORD *)(v7 + 24);
                  goto LABEL_16;
                }
                sub_391E830(v73, v19);
                v105 = *(_DWORD *)(a1 + 184);
                if ( v105 )
                {
                  v106 = *(_QWORD *)(v7 + 8);
                  v107 = v105 - 1;
                  v108 = 1;
                  v104 = 0;
                  v109 = *(_QWORD *)(a1 + 168);
                  v110 = v107 & (((unsigned int)v106 >> 9) ^ ((unsigned int)v106 >> 4));
                  v74 = *(_DWORD *)(a1 + 176) + 1;
                  v71 = (__int64 *)(v109 + 16LL * v110);
                  v111 = *v71;
                  if ( v106 == *v71 )
                    goto LABEL_94;
                  while ( v111 != -8 )
                  {
                    if ( v111 == -16 && !v104 )
                      v104 = v71;
                    v110 = v107 & (v108 + v110);
                    v71 = (__int64 *)(v109 + 16LL * v110);
                    v111 = *v71;
                    if ( v106 == *v71 )
                      goto LABEL_94;
                    ++v108;
                  }
                  goto LABEL_130;
                }
              }
              ++*(_DWORD *)(a1 + 176);
              BUG();
            }
            if ( v71 || v25 != (_BYTE *)-16LL )
              v24 = v71;
            v112 = v70 + 1;
            v23 = v21 & (v70 + v23);
            v113 = v22 + 16LL * v23;
            v25 = *(_BYTE **)v113;
            if ( v20 != *(_BYTE **)v113 )
            {
              v70 = v112;
              v71 = v24;
              v24 = (__int64 *)(v22 + 16LL * v23);
              continue;
            }
            break;
          }
          v11 = *(_DWORD *)(v113 + 8);
          goto LABEL_16;
        case 1:
        case 2:
          v12 = *(_QWORD *)(v7 + 8);
          if ( (*(_BYTE *)(v12 + 9) & 0xC) == 8 )
          {
            v13 = *(_QWORD *)(v12 + 24);
            *(_BYTE *)(v12 + 8) |= 4u;
            v12 = *(_QWORD *)(v13 + 24);
          }
          v14 = *(_DWORD *)(a1 + 152);
          if ( v14 )
          {
            v15 = *(_QWORD *)(a1 + 136);
            v16 = (v14 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
            v17 = (__int64 *)(v15 + 16LL * v16);
            v18 = *v17;
            if ( v12 == *v17 )
            {
              v11 = *((_DWORD *)v17 + 2);
              goto LABEL_12;
            }
            v118 = 1;
            v60 = 0;
            while ( v18 != -8 )
            {
              if ( v18 != -16 || v60 )
                v17 = v60;
              v16 = (v14 - 1) & (v118 + v16);
              v115 = (__int64 *)(v15 + 16LL * v16);
              v18 = *v115;
              if ( v12 == *v115 )
              {
                v11 = *((_DWORD *)v115 + 2);
                goto LABEL_12;
              }
              ++v118;
              v60 = v17;
              v17 = (__int64 *)(v15 + 16LL * v16);
            }
            if ( !v60 )
              v60 = v17;
            v61 = *(_DWORD *)(a1 + 144);
            ++*(_QWORD *)(a1 + 128);
            v62 = v61 + 1;
            if ( 4 * v62 < 3 * v14 )
            {
              if ( v14 - *(_DWORD *)(a1 + 148) - v62 > v14 >> 3 )
              {
LABEL_78:
                *(_DWORD *)(a1 + 144) = v62;
                if ( *v60 != -8 )
                  --*(_DWORD *)(a1 + 148);
                *v60 = v12;
                v11 = 0;
                *((_DWORD *)v60 + 2) = 0;
                goto LABEL_12;
              }
              v121 = ((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4);
              sub_391E830(a1 + 128, v14);
              v91 = *(_DWORD *)(a1 + 152);
              if ( v91 )
              {
                v92 = v91 - 1;
                v93 = *(_QWORD *)(a1 + 136);
                v82 = 0;
                v94 = 1;
                v95 = v92 & v121;
                v62 = *(_DWORD *)(a1 + 144) + 1;
                v60 = (__int64 *)(v93 + 16LL * (v92 & v121));
                v96 = *v60;
                if ( v12 == *v60 )
                  goto LABEL_78;
                while ( v96 != -8 )
                {
                  if ( v96 == -16 && !v82 )
                    v82 = v60;
                  v95 = v92 & (v94 + v95);
                  v60 = (__int64 *)(v93 + 16LL * v95);
                  v96 = *v60;
                  if ( v12 == *v60 )
                    goto LABEL_78;
                  ++v94;
                }
                goto LABEL_103;
              }
              goto LABEL_25;
            }
          }
          else
          {
            ++*(_QWORD *)(a1 + 128);
          }
          sub_391E830(a1 + 128, 2 * v14);
          v76 = *(_DWORD *)(a1 + 152);
          if ( v76 )
          {
            v77 = v76 - 1;
            v78 = *(_QWORD *)(a1 + 136);
            v79 = v77 & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
            v62 = *(_DWORD *)(a1 + 144) + 1;
            v60 = (__int64 *)(v78 + 16LL * v79);
            v80 = *v60;
            if ( v12 == *v60 )
              goto LABEL_78;
            v81 = 1;
            v82 = 0;
            while ( v80 != -8 )
            {
              if ( v80 == -16 && !v82 )
                v82 = v60;
              v79 = v77 & (v81 + v79);
              v60 = (__int64 *)(v78 + 16LL * v79);
              v80 = *v60;
              if ( v12 == *v60 )
                goto LABEL_78;
              ++v81;
            }
LABEL_103:
            if ( v82 )
              v60 = v82;
            goto LABEL_78;
          }
LABEL_25:
          ++*(_DWORD *)(a1 + 144);
          BUG();
        case 3:
        case 4:
        case 5:
          v10 = *(_QWORD *)(v7 + 8);
          if ( (*(_BYTE *)(v10 + 9) & 0xC) != 8 )
          {
            if ( (*(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL) == 0 )
            {
              v11 = 0;
              goto LABEL_16;
            }
LABEL_47:
            v44 = *(_DWORD *)(a1 + 216);
            if ( v44 )
            {
              v45 = *(_QWORD *)(a1 + 200);
              v46 = (v44 - 1) & (((unsigned int)v10 >> 4) ^ ((unsigned int)v10 >> 9));
              v47 = (__int64 *)(v45 + 24LL * v46);
              v48 = *v47;
              if ( v10 == *v47 )
              {
                v49 = *((_DWORD *)v47 + 3);
                v50 = (unsigned __int64)*((unsigned int *)v47 + 2) << 6;
                goto LABEL_50;
              }
              v119 = 1;
              v68 = 0;
              while ( v48 != -8 )
              {
                if ( v48 != -16 || v68 )
                  v47 = v68;
                v46 = (v44 - 1) & (v119 + v46);
                v116 = (__int64 *)(v45 + 24LL * v46);
                v48 = *v116;
                if ( v10 == *v116 )
                {
                  v49 = *((_DWORD *)v116 + 3);
                  v50 = (unsigned __int64)*((unsigned int *)v116 + 2) << 6;
                  goto LABEL_50;
                }
                ++v119;
                v68 = v47;
                v47 = (__int64 *)(v45 + 24LL * v46);
              }
              v83 = *(_DWORD *)(a1 + 208);
              if ( !v68 )
                v68 = v47;
              ++*(_QWORD *)(a1 + 192);
              v67 = v83 + 1;
              if ( 4 * (v83 + 1) < 3 * v44 )
              {
                if ( v44 - *(_DWORD *)(a1 + 212) - v67 > v44 >> 3 )
                  goto LABEL_84;
                v120 = ((unsigned int)v10 >> 4) ^ ((unsigned int)v10 >> 9);
                sub_39205F0(a1 + 192, v44);
                v84 = *(_DWORD *)(a1 + 216);
                if ( v84 )
                {
                  v85 = v84 - 1;
                  v86 = *(_QWORD *)(a1 + 200);
                  v87 = 1;
                  v88 = v85 & v120;
                  v67 = *(_DWORD *)(a1 + 208) + 1;
                  v89 = 0;
                  v68 = (__int64 *)(v86 + 24LL * (v85 & v120));
                  v90 = *v68;
                  if ( *v68 != v10 )
                  {
                    while ( v90 != -8 )
                    {
                      if ( !v89 && v90 == -16 )
                        v89 = v68;
                      v88 = v85 & (v87 + v88);
                      v68 = (__int64 *)(v86 + 24LL * v88);
                      v90 = *v68;
                      if ( v10 == *v68 )
                        goto LABEL_84;
                      ++v87;
                    }
                    goto LABEL_115;
                  }
                  goto LABEL_84;
                }
                goto LABEL_27;
              }
            }
            else
            {
              ++*(_QWORD *)(a1 + 192);
            }
            sub_39205F0(a1 + 192, 2 * v44);
            v63 = *(_DWORD *)(a1 + 216);
            if ( v63 )
            {
              v64 = v63 - 1;
              v65 = *(_QWORD *)(a1 + 200);
              v66 = v64 & (((unsigned int)v10 >> 9) ^ ((unsigned int)v10 >> 4));
              v67 = *(_DWORD *)(a1 + 208) + 1;
              v68 = (__int64 *)(v65 + 24LL * v66);
              v69 = *v68;
              if ( v10 != *v68 )
              {
                v114 = 1;
                v89 = 0;
                while ( v69 != -8 )
                {
                  if ( !v89 && v69 == -16 )
                    v89 = v68;
                  v66 = v64 & (v114 + v66);
                  v68 = (__int64 *)(v65 + 24LL * v66);
                  v69 = *v68;
                  if ( v10 == *v68 )
                    goto LABEL_84;
                  ++v114;
                }
LABEL_115:
                if ( v89 )
                  v68 = v89;
              }
LABEL_84:
              *(_DWORD *)(a1 + 208) = v67;
              if ( *v68 != -8 )
                --*(_DWORD *)(a1 + 212);
              *v68 = v10;
              v50 = 0;
              v68[1] = 0;
              *((_DWORD *)v68 + 4) = 0;
              v49 = 0;
LABEL_50:
              v11 = *(_DWORD *)(*(_QWORD *)(a1 + 696) + v50 + 24) + *(_DWORD *)(v7 + 16) + v49;
              goto LABEL_12;
            }
LABEL_27:
            ++*(_DWORD *)(a1 + 208);
            BUG();
          }
          v42 = *(_QWORD *)(v10 + 24);
          *(_BYTE *)(v10 + 8) |= 4u;
          v10 = *(_QWORD *)(v42 + 24);
          if ( (*(_QWORD *)v10 & 0xFFFFFFFFFFFFFFF8LL) != 0 )
            goto LABEL_47;
          if ( (*(_BYTE *)(v10 + 9) & 0xC) == 8 )
          {
            *(_BYTE *)(v10 + 8) |= 4u;
            v43 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v10 + 24));
            v11 = v43;
            *(_QWORD *)v10 = v43 | *(_QWORD *)v10 & 7LL;
            if ( v43 )
              goto LABEL_47;
LABEL_12:
            v9 = *(_DWORD *)(v7 + 24);
            goto LABEL_16;
          }
          v9 = *(_DWORD *)(v7 + 24);
          v11 = 0;
LABEL_16:
          switch ( v9 )
          {
            case 0:
            case 3:
            case 6:
            case 7:
              v26 = 1;
              v27 = v11 & 0x7F;
              v28 = (unsigned __int64)(unsigned int)v11 >> 7;
              v29 = v125;
              v30 = 1;
              while ( 2 )
              {
                v31 = v29 + 1;
                *v29 = v27 | 0x80;
                if ( v28 )
                {
                  ++v30;
                  v27 = v28 & 0x7F;
                  LOBYTE(v26) = v30 <= 4;
                  v28 >>= 7;
                  if ( v28 || v30 <= 4 )
                  {
                    ++v29;
                    continue;
                  }
                  *v31 = v27;
                  v32 = (_DWORD)v29 + 2;
                }
                else if ( (_BYTE)v26 )
                {
                  if ( v30 != 4 )
                    v31 = (char *)memset(v31, 128, 4 - v30) + 4 - v30;
                  *v31 = 0;
                  v32 = (_DWORD)v31 + 1;
                }
                else
                {
                  v32 = (_DWORD)v29 + 1;
                }
                break;
              }
              result = (*(__int64 (__fastcall **)(__int64, _QWORD *, _QWORD, __int64, __int64))(*(_QWORD *)v5 + 88LL))(
                         v5,
                         v125,
                         v32 - (unsigned int)v125,
                         v8,
                         v26);
              goto LABEL_23;
            case 1:
            case 4:
              v35 = 0;
              v36 = v125;
              v37 = v11 & 0x7F;
              v38 = (__int64)v11 >> 7;
              if ( !v38 )
                goto LABEL_36;
              break;
            case 2:
            case 5:
            case 8:
            case 9:
              LODWORD(v125[0]) = v11;
              result = (*(__int64 (__fastcall **)(__int64, _QWORD *, __int64, __int64))(*(_QWORD *)v5 + 88LL))(
                         v5,
                         v125,
                         4,
                         v8);
              goto LABEL_23;
            default:
              goto LABEL_28;
          }
          while ( v38 != -1 || (v11 & 0x40) == 0 )
          {
            while ( 1 )
            {
              ++v36;
              ++v35;
              LOBYTE(v11) = v38;
              *(v36 - 1) = v37 | 0x80;
              v37 = v38 & 0x7F;
              v38 >>= 7;
              if ( v38 )
                break;
LABEL_36:
              v39 = v11 & 0x40;
              if ( !v39 )
              {
                v40 = v35 + 1;
                v41 = v36 + 1;
                if ( (unsigned int)(v35 + 1) > 4 )
                  goto LABEL_41;
                *v36 = v37 | 0x80;
                v51 = v36 + 1;
                goto LABEL_54;
              }
            }
          }
          v40 = v35 + 1;
          v41 = v36 + 1;
          if ( (unsigned int)(v35 + 1) <= 4 )
          {
            LOBYTE(v39) = 127;
            *v36 = v37 | 0x80;
            v51 = v36 + 1;
LABEL_54:
            if ( v40 != 4 )
            {
              v52 = (unsigned int)(3 - v35);
              if ( (unsigned int)(v35 + 2) > 4 )
                v52 = 1;
              if ( (_DWORD)v52 )
              {
                v53 = 0;
                do
                {
                  v54 = v53++;
                  v41[v54] = v39 | 0x80;
                }
                while ( v53 < (unsigned int)v52 );
              }
              v51 = &v41[v52];
            }
            *v51 = v39;
            LODWORD(v41) = (_DWORD)v51 + 1;
          }
          else
          {
LABEL_41:
            *v36 = v37;
          }
          result = (*(__int64 (__fastcall **)(__int64, _QWORD *, _QWORD, __int64))(*(_QWORD *)v5 + 88LL))(
                     v5,
                     v125,
                     (unsigned int)v41 - (unsigned int)v125,
                     v8);
LABEL_23:
          v7 += 40;
          if ( v122 == v7 )
            return result;
          continue;
        case 6:
          v11 = sub_391F690(a1, v7);
          v9 = *(_DWORD *)(v7 + 24);
          goto LABEL_16;
        case 8:
        case 9:
          v33 = *(_QWORD *)(v7 + 8);
          v34 = *(_QWORD *)v33 & 0xFFFFFFFFFFFFFFF8LL;
          if ( !v34 )
          {
            if ( (*(_BYTE *)(v33 + 9) & 0xC) != 8
              || (*(_BYTE *)(v33 + 8) |= 4u,
                  v34 = (unsigned __int64)sub_38CE440(*(_QWORD *)(v33 + 24)),
                  *(_QWORD *)v33 = v34 | *(_QWORD *)v33 & 7LL,
                  !v34) )
            {
LABEL_28:
              BUG();
            }
            v9 = *(_DWORD *)(v7 + 24);
          }
          v11 = *(_DWORD *)(*(_QWORD *)(v34 + 24) + 184LL) + *(_QWORD *)(v7 + 16);
          goto LABEL_16;
      }
    }
  }
  return result;
}
