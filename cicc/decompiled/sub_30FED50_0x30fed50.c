// Function: sub_30FED50
// Address: 0x30fed50
//
void __fastcall sub_30FED50(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4, __int64 a5)
{
  __int64 v8; // rax
  void (__fastcall *v9)(__int64, __int64, __int64); // rax
  __int64 v10; // rax
  int v11; // eax
  __int64 v12; // rax
  __int64 v13; // rdi
  unsigned __int64 v14; // rcx
  unsigned __int64 v15; // r14
  __int64 v16; // rbx
  __int64 v17; // r10
  __int64 v18; // rbx
  __int64 i; // r12
  __int64 v20; // r14
  unsigned __int8 *v21; // rdi
  unsigned __int8 *v22; // rax
  __int64 v23; // r13
  __int64 v24; // r11
  unsigned int v25; // esi
  __int64 v26; // r9
  unsigned int v27; // r8d
  __int64 *v28; // rax
  __int64 v29; // rdi
  unsigned __int64 v30; // rsi
  unsigned __int64 *v31; // rdx
  _QWORD *v32; // rax
  _QWORD *v33; // rdi
  __int64 v34; // rcx
  __int64 v35; // rdx
  unsigned int v36; // eax
  __int64 v37; // rax
  unsigned __int64 v38; // rbx
  __int64 v39; // r13
  __int64 v40; // r12
  __int64 v41; // r14
  unsigned int v42; // esi
  __int64 v43; // r8
  __int64 v44; // rcx
  _QWORD *v45; // rax
  __int64 v46; // rdi
  unsigned __int64 v47; // rcx
  unsigned __int64 *v48; // rdx
  __int64 v49; // r12
  _QWORD *v50; // rax
  __int64 v51; // rsi
  __int64 v52; // rdx
  __int64 v53; // rax
  __int64 v54; // rsi
  _QWORD *v55; // rax
  _QWORD *v56; // rdx
  char v57; // di
  __int64 v58; // r12
  __int64 v59; // r8
  __int64 v60; // rdi
  _QWORD *v61; // rdx
  __int64 v62; // rcx
  unsigned int v63; // esi
  __int64 v64; // r13
  int v65; // r8d
  int v66; // r8d
  __int64 v67; // r9
  int v68; // edx
  unsigned int v69; // ecx
  _QWORD *v70; // rax
  __int64 v71; // r11
  int v72; // edi
  _QWORD *v73; // rsi
  __int64 *v74; // rdx
  int v75; // eax
  int v76; // eax
  __int64 *v77; // rdx
  int v78; // eax
  int v79; // eax
  int v80; // esi
  int v81; // esi
  __int64 v82; // r9
  unsigned int v83; // ecx
  __int64 v84; // rdi
  int v85; // r10d
  __int64 *v86; // r8
  int v87; // esi
  int v88; // esi
  __int64 v89; // r9
  int v90; // r10d
  unsigned int v91; // ecx
  __int64 v92; // rdi
  unsigned __int64 v93; // rdi
  int v94; // r11d
  int v95; // r11d
  __int64 v96; // r9
  unsigned int v97; // ecx
  __int64 v98; // r8
  int v99; // edi
  __int64 *v100; // rsi
  int v101; // eax
  __int64 v102; // r9
  int v103; // edi
  __int64 v104; // rcx
  __int64 v105; // r8
  int v106; // r10d
  int v107; // edx
  int v108; // r8d
  int v109; // r8d
  __int64 v110; // r9
  _QWORD *v111; // rdi
  unsigned int v112; // r15d
  int v113; // ecx
  __int64 v114; // rsi
  __int64 v115; // [rsp+8h] [rbp-148h]
  unsigned int v116; // [rsp+10h] [rbp-140h]
  int v117; // [rsp+18h] [rbp-138h]
  __int64 v118; // [rsp+18h] [rbp-138h]
  __int64 v119; // [rsp+18h] [rbp-138h]
  unsigned __int64 v120; // [rsp+30h] [rbp-120h]
  __int64 v121; // [rsp+38h] [rbp-118h]
  __int64 v122; // [rsp+40h] [rbp-110h]
  unsigned __int64 v123; // [rsp+40h] [rbp-110h]
  int v124; // [rsp+40h] [rbp-110h]
  int v125; // [rsp+40h] [rbp-110h]
  unsigned int v126; // [rsp+4Ch] [rbp-104h]
  _QWORD *v127; // [rsp+50h] [rbp-100h]
  _BYTE v129[56]; // [rsp+60h] [rbp-F0h] BYREF
  __int64 v130; // [rsp+98h] [rbp-B8h]
  __int64 v131; // [rsp+B0h] [rbp-A0h] BYREF
  __int64 v132; // [rsp+B8h] [rbp-98h]
  __int64 v133; // [rsp+C0h] [rbp-90h]
  __int64 v134; // [rsp+C8h] [rbp-88h]
  __int64 v135; // [rsp+D0h] [rbp-80h]
  unsigned __int64 v136; // [rsp+D8h] [rbp-78h]
  __int64 v137; // [rsp+E0h] [rbp-70h]
  __int64 v138; // [rsp+E8h] [rbp-68h]
  unsigned __int64 v139; // [rsp+F0h] [rbp-60h]
  __int64 v140; // [rsp+F8h] [rbp-58h]
  __int64 v141; // [rsp+100h] [rbp-50h]
  unsigned __int64 v142; // [rsp+108h] [rbp-48h]
  __int64 v143; // [rsp+110h] [rbp-40h]
  __int64 v144; // [rsp+118h] [rbp-38h]

  LOBYTE(v132) = 0;
  v8 = sub_BC0510(a3, &unk_4F82418, a2);
  sub_30CBEF0(a1, a2, *(_QWORD *)(v8 + 8), v131, v132);
  *(_QWORD *)a1 = &unk_4A32870;
  *(_QWORD *)(a1 + 80) = *a4;
  *a4 = 0;
  *(_QWORD *)(a1 + 104) = 0;
  v9 = *(void (__fastcall **)(__int64, __int64, __int64))(a5 + 16);
  if ( v9 )
  {
    v9(a1 + 88, a5, 2);
    *(_QWORD *)(a1 + 112) = *(_QWORD *)(a5 + 24);
    *(_QWORD *)(a1 + 104) = *(_QWORD *)(a5 + 16);
  }
  *(_DWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 144) = a1 + 128;
  *(_QWORD *)(a1 + 152) = a1 + 128;
  *(_QWORD *)(a1 + 160) = 0;
  v10 = sub_BC0510(a3, qword_4F86C48, a2);
  *(_QWORD *)(a1 + 176) = 0;
  *(_QWORD *)(a1 + 224) = a1 + 208;
  *(_QWORD *)(a1 + 232) = a1 + 208;
  *(_QWORD *)(a1 + 168) = v10 + 8;
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 192) = 0;
  *(_DWORD *)(a1 + 208) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 240) = 0;
  v127 = (_QWORD *)(a1 + 208);
  v11 = sub_30FCEE0(a1);
  *(_DWORD *)(a1 + 248) = v11;
  *(_DWORD *)(a1 + 252) = v11;
  *(_QWORD *)(a1 + 256) = 0;
  *(_QWORD *)(a1 + 264) = a1 + 288;
  *(_QWORD *)(a1 + 272) = 1;
  *(_DWORD *)(a1 + 280) = 0;
  *(_BYTE *)(a1 + 284) = 1;
  *(_QWORD *)(a1 + 296) = 0;
  *(_QWORD *)(a1 + 304) = 0;
  *(_QWORD *)(a1 + 312) = 0;
  *(_DWORD *)(a1 + 320) = 0;
  *(_QWORD *)(a1 + 328) = 0;
  *(_QWORD *)(a1 + 336) = 0;
  *(_QWORD *)(a1 + 344) = 0;
  *(_DWORD *)(a1 + 352) = 0;
  *(_BYTE *)(a1 + 360) = 0;
  v115 = a1 + 296;
  v12 = sub_BC0510(a3, &unk_4F87C68, a2);
  v13 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 368) = v12 + 8;
  (*(void (__fastcall **)(__int64, const char *, _QWORD))(*(_QWORD *)v13 + 16LL))(v13, byte_3F871B3, 0);
  sub_D12090((__int64)v129, a2);
  v131 = 0;
  v132 = 0;
  v133 = 0;
  v134 = 0;
  v135 = 0;
  v136 = 0;
  v137 = 0;
  v138 = 0;
  v139 = 0;
  v140 = 0;
  v141 = 0;
  v142 = 0;
  v143 = 0;
  v144 = 0;
  sub_D126D0((__int64)&v131, v130);
  sub_D12BD0((__int64)&v131);
  v14 = v139;
  v122 = v140;
  if ( v140 != v139 )
  {
    while ( 1 )
    {
      v126 = 0;
      v15 = v14;
      do
      {
        while ( 1 )
        {
          v16 = *(_QWORD *)(*(_QWORD *)v15 + 8LL);
          if ( v16 )
          {
            if ( !sub_B2FC80(*(_QWORD *)(*(_QWORD *)v15 + 8LL)) )
              break;
          }
          v15 += 8LL;
          if ( v122 == v15 )
            goto LABEL_44;
        }
        v17 = v16 + 72;
        v18 = *(_QWORD *)(v16 + 80);
        if ( v17 == v18 )
        {
          v120 = v15;
          i = 0;
          v20 = v17;
        }
        else
        {
          if ( !v18 )
            BUG();
          while ( 1 )
          {
            i = *(_QWORD *)(v18 + 32);
            if ( i != v18 + 24 )
              break;
            v18 = *(_QWORD *)(v18 + 8);
            if ( v17 == v18 )
              break;
            if ( !v18 )
              BUG();
          }
          v120 = v15;
          v20 = v17;
        }
        while ( v18 != v20 )
        {
          v21 = (unsigned __int8 *)(i - 24);
          if ( !i )
            v21 = 0;
          v22 = sub_30FB990(v21);
          if ( !v22 )
            goto LABEL_36;
          v23 = *((_QWORD *)v22 - 4);
          if ( v23 )
          {
            if ( *(_BYTE *)v23 )
            {
              v23 = 0;
            }
            else if ( *(_QWORD *)(v23 + 24) != *((_QWORD *)v22 + 10) )
            {
              v23 = 0;
            }
          }
          v24 = *(_QWORD *)(a1 + 168);
          v25 = *(_DWORD *)(v24 + 120);
          v121 = v24 + 96;
          if ( !v25 )
          {
            ++*(_QWORD *)(v24 + 96);
            goto LABEL_108;
          }
          v26 = *(_QWORD *)(v24 + 104);
          v27 = (v25 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v28 = (__int64 *)(v26 + 16LL * v27);
          v29 = *v28;
          if ( v23 == *v28 )
            goto LABEL_25;
          v117 = 1;
          v74 = 0;
          while ( 1 )
          {
            if ( v29 == -4096 )
            {
              if ( !v74 )
                v74 = v28;
              v75 = *(_DWORD *)(v24 + 112);
              ++*(_QWORD *)(v24 + 96);
              v76 = v75 + 1;
              if ( 4 * v76 >= 3 * v25 )
              {
LABEL_108:
                v118 = v24;
                sub_D25040(v121, 2 * v25);
                v24 = v118;
                v80 = *(_DWORD *)(v118 + 120);
                if ( v80 )
                {
                  v81 = v80 - 1;
                  v82 = *(_QWORD *)(v118 + 104);
                  v83 = v81 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
                  v76 = *(_DWORD *)(v118 + 112) + 1;
                  v74 = (__int64 *)(v82 + 16LL * v83);
                  v84 = *v74;
                  if ( v23 == *v74 )
                    goto LABEL_91;
                  v85 = 1;
                  v86 = 0;
                  while ( v84 != -4096 )
                  {
                    if ( v84 == -8192 && !v86 )
                      v86 = v74;
                    v83 = v81 & (v85 + v83);
                    v74 = (__int64 *)(v82 + 16LL * v83);
                    v84 = *v74;
                    if ( v23 == *v74 )
                      goto LABEL_91;
                    ++v85;
                  }
LABEL_120:
                  if ( v86 )
                    v74 = v86;
                  goto LABEL_91;
                }
              }
              else
              {
                if ( v25 - *(_DWORD *)(v24 + 116) - v76 > v25 >> 3 )
                {
LABEL_91:
                  *(_DWORD *)(v24 + 112) = v76;
                  if ( *v74 != -4096 )
                    --*(_DWORD *)(v24 + 116);
                  *v74 = v23;
                  v31 = (unsigned __int64 *)(v74 + 1);
                  *v31 = 0;
LABEL_94:
                  v30 = sub_D28F90((__int64 *)v24, v23, v31);
                  goto LABEL_26;
                }
                v119 = v24;
                v116 = ((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4);
                sub_D25040(v121, v25);
                v24 = v119;
                v87 = *(_DWORD *)(v119 + 120);
                if ( v87 )
                {
                  v88 = v87 - 1;
                  v89 = *(_QWORD *)(v119 + 104);
                  v86 = 0;
                  v90 = 1;
                  v91 = v88 & v116;
                  v76 = *(_DWORD *)(v119 + 112) + 1;
                  v74 = (__int64 *)(v89 + 16LL * (v88 & v116));
                  v92 = *v74;
                  if ( v23 == *v74 )
                    goto LABEL_91;
                  while ( v92 != -4096 )
                  {
                    if ( !v86 && v92 == -8192 )
                      v86 = v74;
                    v91 = v88 & (v90 + v91);
                    v74 = (__int64 *)(v89 + 16LL * v91);
                    v92 = *v74;
                    if ( v23 == *v74 )
                      goto LABEL_91;
                    ++v90;
                  }
                  goto LABEL_120;
                }
              }
              ++*(_DWORD *)(v24 + 112);
              BUG();
            }
            if ( v74 || v29 != -8192 )
              v28 = v74;
            v27 = (v25 - 1) & (v117 + v27);
            v29 = *(_QWORD *)(v26 + 16LL * v27);
            if ( v23 == v29 )
              break;
            ++v117;
            v74 = v28;
            v28 = (__int64 *)(v26 + 16LL * v27);
          }
          v28 = (__int64 *)(v26 + 16LL * v27);
LABEL_25:
          v30 = v28[1];
          v31 = (unsigned __int64 *)(v28 + 1);
          if ( !v30 )
            goto LABEL_94;
LABEL_26:
          v32 = *(_QWORD **)(a1 + 216);
          if ( v32 )
          {
            v33 = v127;
            do
            {
              while ( 1 )
              {
                v34 = v32[2];
                v35 = v32[3];
                if ( v32[4] >= v30 )
                  break;
                v32 = (_QWORD *)v32[3];
                if ( !v35 )
                  goto LABEL_31;
              }
              v33 = v32;
              v32 = (_QWORD *)v32[2];
            }
            while ( v34 );
LABEL_31:
            if ( v127 != v33 && v33[4] <= v30 )
            {
              v36 = *((_DWORD *)v33 + 10) + 1;
              if ( v126 >= v36 )
                v36 = v126;
              v126 = v36;
            }
          }
LABEL_36:
          for ( i = *(_QWORD *)(i + 8); ; i = *(_QWORD *)(v18 + 32) )
          {
            v37 = v18 - 24;
            if ( !v18 )
              v37 = 0;
            if ( i != v37 + 48 )
              break;
            v18 = *(_QWORD *)(v18 + 8);
            if ( v20 == v18 )
              break;
            if ( !v18 )
              BUG();
          }
        }
        v15 = v120 + 8;
      }
      while ( v122 != v120 + 8 );
LABEL_44:
      v38 = v139;
      v39 = v140;
      if ( v140 != v139 )
        break;
LABEL_65:
      sub_D12BD0((__int64)&v131);
      v14 = v139;
      v122 = v140;
      if ( v139 == v140 )
        goto LABEL_66;
    }
    while ( 1 )
    {
      while ( 1 )
      {
        v40 = *(_QWORD *)(*(_QWORD *)v38 + 8LL);
        if ( v40 )
        {
          if ( !sub_B2FC80(*(_QWORD *)(*(_QWORD *)v38 + 8LL)) )
            break;
        }
        v38 += 8LL;
        if ( v39 == v38 )
          goto LABEL_65;
      }
      v41 = *(_QWORD *)(a1 + 168);
      v42 = *(_DWORD *)(v41 + 120);
      if ( v42 )
      {
        v43 = *(_QWORD *)(v41 + 104);
        v44 = (v42 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
        v45 = (_QWORD *)(v43 + 16 * v44);
        v46 = *v45;
        if ( v40 == *v45 )
        {
LABEL_51:
          v47 = v45[1];
          v48 = v45 + 1;
          if ( v47 )
            goto LABEL_52;
          goto LABEL_106;
        }
        v124 = 1;
        v77 = 0;
        while ( v46 != -4096 )
        {
          if ( !v77 && v46 == -8192 )
            v77 = v45;
          LODWORD(v44) = (v42 - 1) & (v124 + v44);
          v45 = (_QWORD *)(v43 + 16LL * (unsigned int)v44);
          v46 = *v45;
          if ( v40 == *v45 )
            goto LABEL_51;
          ++v124;
        }
        if ( !v77 )
          v77 = v45;
        v78 = *(_DWORD *)(v41 + 112);
        ++*(_QWORD *)(v41 + 96);
        v79 = v78 + 1;
        if ( 4 * v79 < 3 * v42 )
        {
          if ( v42 - *(_DWORD *)(v41 + 116) - v79 > v42 >> 3 )
            goto LABEL_103;
          sub_D25040(v41 + 96, v42);
          v101 = *(_DWORD *)(v41 + 120);
          if ( !v101 )
          {
LABEL_193:
            ++*(_DWORD *)(v41 + 112);
            BUG();
          }
          v102 = *(_QWORD *)(v41 + 104);
          v100 = 0;
          v125 = v101 - 1;
          v103 = 1;
          LODWORD(v104) = (v101 - 1) & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
          v79 = *(_DWORD *)(v41 + 112) + 1;
          v77 = (__int64 *)(v102 + 16LL * (unsigned int)v104);
          v105 = *v77;
          if ( v40 == *v77 )
            goto LABEL_103;
          while ( v105 != -4096 )
          {
            if ( v105 == -8192 && !v100 )
              v100 = v77;
            v104 = v125 & (unsigned int)(v104 + v103);
            v77 = (__int64 *)(v102 + 16 * v104);
            v105 = *v77;
            if ( v40 == *v77 )
              goto LABEL_103;
            ++v103;
          }
          goto LABEL_139;
        }
      }
      else
      {
        ++*(_QWORD *)(v41 + 96);
      }
      sub_D25040(v41 + 96, 2 * v42);
      v94 = *(_DWORD *)(v41 + 120);
      if ( !v94 )
        goto LABEL_193;
      v95 = v94 - 1;
      v96 = *(_QWORD *)(v41 + 104);
      v97 = v95 & (((unsigned int)v40 >> 9) ^ ((unsigned int)v40 >> 4));
      v79 = *(_DWORD *)(v41 + 112) + 1;
      v77 = (__int64 *)(v96 + 16LL * v97);
      v98 = *v77;
      if ( v40 == *v77 )
        goto LABEL_103;
      v99 = 1;
      v100 = 0;
      while ( v98 != -4096 )
      {
        if ( v98 == -8192 && !v100 )
          v100 = v77;
        v97 = v95 & (v99 + v97);
        v77 = (__int64 *)(v96 + 16LL * v97);
        v98 = *v77;
        if ( v40 == *v77 )
          goto LABEL_103;
        ++v99;
      }
LABEL_139:
      if ( v100 )
        v77 = v100;
LABEL_103:
      *(_DWORD *)(v41 + 112) = v79;
      if ( *v77 != -4096 )
        --*(_DWORD *)(v41 + 116);
      *v77 = v40;
      v48 = (unsigned __int64 *)(v77 + 1);
      *v48 = 0;
LABEL_106:
      v47 = sub_D28F90((__int64 *)v41, v40, v48);
LABEL_52:
      v49 = (__int64)v127;
      v50 = *(_QWORD **)(a1 + 216);
      if ( !v50 )
        goto LABEL_59;
      do
      {
        while ( 1 )
        {
          v51 = v50[2];
          v52 = v50[3];
          if ( v50[4] >= v47 )
            break;
          v50 = (_QWORD *)v50[3];
          if ( !v52 )
            goto LABEL_57;
        }
        v49 = (__int64)v50;
        v50 = (_QWORD *)v50[2];
      }
      while ( v51 );
LABEL_57:
      if ( v127 == (_QWORD *)v49 || *(_QWORD *)(v49 + 32) > v47 )
      {
LABEL_59:
        v123 = v47;
        v53 = sub_22077B0(0x30u);
        v54 = v49;
        *(_DWORD *)(v53 + 40) = 0;
        v49 = v53;
        *(_QWORD *)(v53 + 32) = v123;
        v55 = sub_30FDD00((_QWORD *)(a1 + 200), v54, (unsigned __int64 *)(v53 + 32));
        if ( v56 )
        {
          v57 = v127 == v56 || v55 || v123 < v56[4];
          sub_220F040(v57, v49, v56, v127);
          ++*(_QWORD *)(a1 + 240);
        }
        else
        {
          v93 = v49;
          v49 = (__int64)v55;
          j_j___libc_free_0(v93);
        }
      }
      v38 += 8LL;
      *(_DWORD *)(v49 + 40) = v126;
      if ( v39 == v38 )
        goto LABEL_65;
    }
  }
LABEL_66:
  if ( v142 )
    j_j___libc_free_0(v142);
  if ( v139 )
    j_j___libc_free_0(v139);
  if ( v136 )
    j_j___libc_free_0(v136);
  sub_C7D6A0(v133, 16LL * (unsigned int)v135, 8);
  v58 = *(_QWORD *)(a1 + 224);
  if ( v127 != (_QWORD *)v58 )
  {
    while ( 1 )
    {
      v63 = *(_DWORD *)(a1 + 320);
      v64 = *(_QWORD *)(v58 + 32);
      if ( !v63 )
        break;
      v59 = *(_QWORD *)(a1 + 304);
      LODWORD(v60) = (v63 - 1) & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
      v61 = (_QWORD *)(v59 + 8LL * (unsigned int)v60);
      v62 = *v61;
      if ( v64 != *v61 )
      {
        v106 = 1;
        v70 = 0;
        while ( v62 != -4096 )
        {
          if ( !v70 && v62 == -8192 )
            v70 = v61;
          v60 = (v63 - 1) & ((_DWORD)v60 + v106);
          v61 = (_QWORD *)(v59 + 8 * v60);
          v62 = *v61;
          if ( *v61 == v64 )
            goto LABEL_75;
          ++v106;
        }
        if ( !v70 )
          v70 = v61;
        v107 = *(_DWORD *)(a1 + 312);
        ++*(_QWORD *)(a1 + 296);
        v68 = v107 + 1;
        if ( 4 * v68 < 3 * v63 )
        {
          if ( v63 - *(_DWORD *)(a1 + 316) - v68 <= v63 >> 3 )
          {
            sub_30FDE00(v115, v63);
            v108 = *(_DWORD *)(a1 + 320);
            if ( !v108 )
            {
LABEL_194:
              ++*(_DWORD *)(a1 + 312);
              BUG();
            }
            v109 = v108 - 1;
            v110 = *(_QWORD *)(a1 + 304);
            v111 = 0;
            v112 = v109 & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
            v113 = 1;
            v68 = *(_DWORD *)(a1 + 312) + 1;
            v70 = (_QWORD *)(v110 + 8LL * v112);
            v114 = *v70;
            if ( *v70 != v64 )
            {
              while ( v114 != -4096 )
              {
                if ( !v111 && v114 == -8192 )
                  v111 = v70;
                v112 = v109 & (v113 + v112);
                v70 = (_QWORD *)(v110 + 8LL * v112);
                v114 = *v70;
                if ( v64 == *v70 )
                  goto LABEL_148;
                ++v113;
              }
              if ( v111 )
                v70 = v111;
            }
          }
          goto LABEL_148;
        }
LABEL_78:
        sub_30FDE00(v115, 2 * v63);
        v65 = *(_DWORD *)(a1 + 320);
        if ( !v65 )
          goto LABEL_194;
        v66 = v65 - 1;
        v67 = *(_QWORD *)(a1 + 304);
        v68 = *(_DWORD *)(a1 + 312) + 1;
        v69 = v66 & (((unsigned int)v64 >> 9) ^ ((unsigned int)v64 >> 4));
        v70 = (_QWORD *)(v67 + 8LL * v69);
        v71 = *v70;
        if ( *v70 != v64 )
        {
          v72 = 1;
          v73 = 0;
          while ( v71 != -4096 )
          {
            if ( v71 == -8192 && !v73 )
              v73 = v70;
            v69 = v66 & (v72 + v69);
            v70 = (_QWORD *)(v67 + 8LL * v69);
            v71 = *v70;
            if ( v64 == *v70 )
              goto LABEL_148;
            ++v72;
          }
          if ( v73 )
            v70 = v73;
        }
LABEL_148:
        *(_DWORD *)(a1 + 312) = v68;
        if ( *v70 != -4096 )
          --*(_DWORD *)(a1 + 316);
        *v70 = v64;
      }
LABEL_75:
      *(_QWORD *)(a1 + 184) += sub_30FCC90(a1, *(_QWORD *)(v64 + 8));
      v58 = sub_220EEE0(v58);
      if ( v127 == (_QWORD *)v58 )
        goto LABEL_125;
    }
    ++*(_QWORD *)(a1 + 296);
    goto LABEL_78;
  }
LABEL_125:
  *(_QWORD *)(a1 + 176) = *(unsigned int *)(a1 + 312);
  sub_D0FA70((__int64)v129);
}
