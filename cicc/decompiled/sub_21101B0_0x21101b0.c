// Function: sub_21101B0
// Address: 0x21101b0
//
void __fastcall sub_21101B0(__int64 a1)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rcx
  _QWORD *v4; // r13
  int v5; // eax
  unsigned __int64 v6; // r14
  int v7; // eax
  unsigned __int64 v8; // r14
  int v9; // eax
  void *v10; // r12
  unsigned __int64 v11; // r14
  int v12; // eax
  unsigned __int64 v13; // r14
  unsigned int v14; // esi
  __int64 v15; // r9
  __int64 v16; // r8
  __int64 v17; // rdi
  _QWORD *v18; // rax
  unsigned __int64 v19; // rcx
  int v20; // ebx
  unsigned int v21; // r14d
  unsigned int v22; // esi
  int v23; // ecx
  int v24; // r8d
  int v25; // r9d
  unsigned __int64 v26; // r9
  int v27; // r8d
  _BYTE *v28; // rdx
  _DWORD *v29; // rax
  _DWORD *i; // rdx
  int v31; // edx
  unsigned int v32; // ecx
  unsigned int v33; // esi
  __int64 v34; // rdi
  __int64 v35; // r9
  __int64 v36; // r8
  __int64 *v37; // rax
  __int64 v38; // rcx
  __int64 v39; // rdx
  __int64 v40; // rax
  __int64 v41; // r11
  __int64 v42; // rcx
  unsigned int v43; // r8d
  __int64 v44; // rdi
  __int64 v45; // rax
  __int64 *v46; // r12
  __int64 v47; // rbx
  __int64 v48; // rsi
  _BYTE *v49; // r8
  unsigned int v50; // edx
  unsigned int v51; // eax
  _QWORD *v52; // rsi
  char v53; // cl
  __int64 *v54; // r8
  __int64 v55; // r9
  unsigned int v56; // ecx
  unsigned int k; // eax
  _QWORD *v58; // r10
  unsigned int v59; // esi
  __int64 *v60; // rbx
  unsigned int v61; // ebx
  unsigned int j; // ecx
  int v63; // r11d
  _QWORD *v64; // rdx
  int v65; // eax
  int v66; // ecx
  int v67; // r11d
  _QWORD *v68; // rdx
  int v69; // eax
  int v70; // eax
  int v71; // r10d
  int v72; // r10d
  __int64 v73; // rsi
  int v74; // r11d
  _QWORD *v75; // rdi
  int v76; // r10d
  int v77; // r10d
  __int64 v78; // r9
  __int64 v79; // rdi
  __int64 v80; // rax
  int v81; // r11d
  _QWORD *v82; // r8
  int v83; // r10d
  int v84; // r10d
  unsigned int v85; // ebx
  __int64 v86; // rsi
  int v87; // r10d
  int v88; // r10d
  __int64 v89; // r9
  int v90; // r8d
  __int64 v91; // rbx
  __int64 v92; // rdi
  _QWORD *v93; // rax
  __int64 v94; // r10
  void *v95; // rax
  __int64 v96; // rax
  void *v97; // rax
  void *v98; // rax
  __int64 v99; // [rsp-100h] [rbp-100h]
  void *v100; // [rsp-E8h] [rbp-E8h]
  int v101; // [rsp-E8h] [rbp-E8h]
  _QWORD *v102; // [rsp-E0h] [rbp-E0h]
  __int64 v103; // [rsp-D8h] [rbp-D8h]
  unsigned __int64 v104; // [rsp-D8h] [rbp-D8h]
  void *v105; // [rsp-C8h] [rbp-C8h]
  void *v106; // [rsp-C0h] [rbp-C0h]
  void *v107; // [rsp-B8h] [rbp-B8h]
  __int64 v108; // [rsp-B0h] [rbp-B0h]
  char v109; // [rsp-B0h] [rbp-B0h]
  unsigned __int64 v110[2]; // [rsp-A8h] [rbp-A8h] BYREF
  int v111; // [rsp-98h] [rbp-98h]
  unsigned __int64 v112[2]; // [rsp-88h] [rbp-88h] BYREF
  int v113; // [rsp-78h] [rbp-78h]
  _BYTE *v114; // [rsp-68h] [rbp-68h] BYREF
  __int64 v115; // [rsp-60h] [rbp-60h]
  _BYTE v116[88]; // [rsp-58h] [rbp-58h] BYREF

  if ( *(_DWORD *)(a1 + 24) )
  {
    v2 = *(_QWORD **)(a1 + 16);
    v3 = &v2[13 * *(unsigned int *)(a1 + 32)];
    v102 = v3;
    if ( v2 != v3 )
    {
      while ( 1 )
      {
        v4 = v2;
        if ( *v2 != -8 && *v2 != -16 )
          break;
        v2 += 13;
        if ( v3 == v2 )
          return;
      }
      v108 = *v2;
      if ( v2 != v3 )
      {
        v99 = a1 + 80;
        while ( 1 )
        {
          v106 = 0;
          v5 = *((_DWORD *)v4 + 6);
          if ( !v5 )
            goto LABEL_12;
          v6 = 8LL * ((unsigned int)(v5 + 63) >> 6);
          v106 = (void *)malloc(v6);
          if ( v106 )
            goto LABEL_11;
          if ( v6 )
            break;
          v98 = (void *)malloc(1u);
          if ( !v98 )
            break;
          v106 = v98;
          memcpy(v98, (const void *)v4[1], 0);
LABEL_12:
          v107 = 0;
          v7 = *((_DWORD *)v4 + 12);
          if ( !v7 )
            goto LABEL_15;
          v8 = 8LL * ((unsigned int)(v7 + 63) >> 6);
          v107 = (void *)malloc(v8);
          if ( v107 )
            goto LABEL_14;
          if ( v8 || (v95 = (void *)malloc(1u)) == 0 )
          {
            sub_16BD1C0("Allocation failed", 1u);
LABEL_14:
            memcpy(v107, (const void *)v4[4], v8);
            goto LABEL_15;
          }
          v107 = v95;
          memcpy(v95, (const void *)v4[4], 0);
LABEL_15:
          v9 = *((_DWORD *)v4 + 18);
          v10 = 0;
          if ( v9 )
          {
            v11 = 8LL * ((unsigned int)(v9 + 63) >> 6);
            v10 = (void *)malloc(v11);
            if ( !v10 )
            {
              if ( v11 || (v96 = malloc(1u)) == 0 )
                sub_16BD1C0("Allocation failed", 1u);
              else
                v10 = (void *)v96;
            }
            memcpy(v10, (const void *)v4[7], v11);
          }
          v105 = 0;
          v12 = *((_DWORD *)v4 + 24);
          if ( v12 )
          {
            v13 = 8LL * ((unsigned int)(v12 + 63) >> 6);
            v105 = (void *)malloc(v13);
            if ( v105 )
              goto LABEL_20;
            if ( v13 || (v97 = (void *)malloc(1u)) == 0 )
            {
              sub_16BD1C0("Allocation failed", 1u);
LABEL_20:
              memcpy(v105, (const void *)v4[10], v13);
              goto LABEL_21;
            }
            v105 = v97;
            memcpy(v97, (const void *)v4[10], 0);
          }
LABEL_21:
          v14 = *(_DWORD *)(a1 + 104);
          if ( !v14 )
          {
            ++*(_QWORD *)(a1 + 80);
LABEL_96:
            sub_19F7080(v99, 2 * v14);
            v71 = *(_DWORD *)(a1 + 104);
            if ( !v71 )
              goto LABEL_171;
            v72 = v71 - 1;
            v15 = *(_QWORD *)(a1 + 88);
            v19 = v72 & (((unsigned int)v108 >> 9) ^ ((unsigned int)v108 >> 4));
            v70 = *(_DWORD *)(a1 + 96) + 1;
            v68 = (_QWORD *)(v15 + 16 * v19);
            v73 = *v68;
            if ( v108 != *v68 )
            {
              v74 = 1;
              v75 = 0;
              while ( v73 != -8 )
              {
                if ( !v75 && v73 == -16 )
                  v75 = v68;
                LODWORD(v16) = v74 + 1;
                LODWORD(v19) = v72 & (v74 + v19);
                v68 = (_QWORD *)(v15 + 16LL * (unsigned int)v19);
                v73 = *v68;
                if ( *v68 == v108 )
                  goto LABEL_92;
                ++v74;
              }
              if ( v75 )
                v68 = v75;
            }
            goto LABEL_92;
          }
          LODWORD(v15) = v14 - 1;
          v16 = *(_QWORD *)(a1 + 88);
          v17 = (v14 - 1) & (((unsigned int)v108 >> 9) ^ ((unsigned int)v108 >> 4));
          v18 = (_QWORD *)(v16 + 16 * v17);
          v19 = *v18;
          if ( v108 == *v18 )
          {
            v20 = *((_DWORD *)v18 + 2);
            v21 = *((_DWORD *)v18 + 3);
            goto LABEL_24;
          }
          v67 = 1;
          v68 = 0;
          while ( v19 != -8 )
          {
            if ( v68 || v19 != -16 )
              v18 = v68;
            LODWORD(v17) = v15 & (v67 + v17);
            v94 = v16 + 16LL * (unsigned int)v17;
            v19 = *(_QWORD *)v94;
            if ( *(_QWORD *)v94 == v108 )
            {
              v20 = *(_DWORD *)(v94 + 8);
              v21 = *(_DWORD *)(v94 + 12);
              goto LABEL_24;
            }
            ++v67;
            v68 = v18;
            v18 = (_QWORD *)(v16 + 16LL * (unsigned int)v17);
          }
          if ( !v68 )
            v68 = v18;
          v69 = *(_DWORD *)(a1 + 96);
          ++*(_QWORD *)(a1 + 80);
          v70 = v69 + 1;
          if ( 4 * v70 >= 3 * v14 )
            goto LABEL_96;
          LODWORD(v19) = v14 - *(_DWORD *)(a1 + 100) - v70;
          if ( (unsigned int)v19 <= v14 >> 3 )
          {
            sub_19F7080(v99, v14);
            v83 = *(_DWORD *)(a1 + 104);
            if ( !v83 )
            {
LABEL_171:
              ++*(_DWORD *)(a1 + 96);
              BUG();
            }
            v84 = v83 - 1;
            v15 = *(_QWORD *)(a1 + 88);
            v19 = 0;
            v85 = v84 & (((unsigned int)v108 >> 9) ^ ((unsigned int)v108 >> 4));
            LODWORD(v16) = 1;
            v70 = *(_DWORD *)(a1 + 96) + 1;
            v68 = (_QWORD *)(v15 + 16LL * v85);
            v86 = *v68;
            if ( v108 != *v68 )
            {
              while ( v86 != -8 )
              {
                if ( v86 == -16 && !v19 )
                  v19 = (unsigned __int64)v68;
                v85 = v84 & (v16 + v85);
                v68 = (_QWORD *)(v15 + 16LL * v85);
                v86 = *v68;
                if ( v108 == *v68 )
                  goto LABEL_92;
                LODWORD(v16) = v16 + 1;
              }
              if ( v19 )
                v68 = (_QWORD *)v19;
            }
          }
LABEL_92:
          *(_DWORD *)(a1 + 96) = v70;
          if ( *v68 != -8 )
            --*(_DWORD *)(a1 + 100);
          v68[1] = 0;
          v21 = 0;
          v20 = 0;
          *v68 = v108;
LABEL_24:
          v22 = *(_DWORD *)(a1 + 128);
          v110[0] = 0;
          v110[1] = 0;
          v111 = 0;
          v112[0] = 0;
          v112[1] = 0;
          v113 = 0;
          sub_13A49F0((__int64)v110, v22, 0, v19, v16, v15);
          sub_13A49F0((__int64)v112, *(_DWORD *)(a1 + 128), 0, v23, v24, v25);
          v26 = *(unsigned int *)(a1 + 128);
          v114 = v116;
          v27 = v26;
          v115 = 0x800000000LL;
          if ( v26 )
          {
            v28 = v116;
            v29 = v116;
            if ( v26 > 8 )
            {
              v101 = v26;
              v104 = v26;
              sub_16CD150((__int64)&v114, v116, v26, 4, v26, v26);
              v28 = v114;
              v27 = v101;
              v26 = v104;
              v29 = &v114[4 * (unsigned int)v115];
            }
            for ( i = &v28[4 * v26]; i != v29; ++v29 )
            {
              if ( v29 )
                *v29 = 0;
            }
            v31 = *(_DWORD *)(a1 + 128);
            LODWORD(v115) = v27;
            if ( v31 )
            {
              v32 = 0;
              do
              {
                if ( (*((_QWORD *)v10 + (v32 >> 6)) & (1LL << v32)) != 0 )
                {
                  *(_QWORD *)(v110[0] + 8LL * (v32 >> 6)) |= 1LL << v32;
                  *(_DWORD *)&v114[4 * v32] = v20;
                }
                ++v32;
              }
              while ( *(_DWORD *)(a1 + 128) > v32 );
            }
          }
          v33 = *(_DWORD *)(a1 + 504);
          v34 = a1 + 480;
          if ( !v33 )
          {
            ++*(_QWORD *)(a1 + 480);
            goto LABEL_104;
          }
          v35 = *(_QWORD *)(a1 + 488);
          v36 = (v33 - 1) & (((unsigned int)v108 >> 4) ^ ((unsigned int)v108 >> 9));
          v37 = (__int64 *)(v35 + 72 * v36);
          v38 = *v37;
          if ( *v37 != v108 )
          {
            v63 = 1;
            v64 = 0;
            while ( v38 != -8 )
            {
              if ( !v64 && v38 == -16 )
                v64 = v37;
              LODWORD(v36) = (v33 - 1) & (v63 + v36);
              v37 = (__int64 *)(v35 + 72LL * (unsigned int)v36);
              v38 = *v37;
              if ( v108 == *v37 )
                goto LABEL_38;
              ++v63;
            }
            if ( !v64 )
              v64 = v37;
            v65 = *(_DWORD *)(a1 + 496);
            ++*(_QWORD *)(a1 + 480);
            v66 = v65 + 1;
            if ( 4 * (v65 + 1) >= 3 * v33 )
            {
LABEL_104:
              sub_210FEC0(v34, 2 * v33);
              v76 = *(_DWORD *)(a1 + 504);
              if ( !v76 )
                goto LABEL_170;
              v77 = v76 - 1;
              v78 = *(_QWORD *)(a1 + 488);
              v66 = *(_DWORD *)(a1 + 496) + 1;
              v79 = v77 & (((unsigned int)v108 >> 9) ^ ((unsigned int)v108 >> 4));
              v64 = (_QWORD *)(v78 + 72 * v79);
              v80 = *v64;
              if ( *v64 != v108 )
              {
                v81 = 1;
                v82 = 0;
                while ( v80 != -8 )
                {
                  if ( v80 == -16 && !v82 )
                    v82 = v64;
                  v79 = v77 & (unsigned int)(v79 + v81);
                  v64 = (_QWORD *)(v78 + 72 * v79);
                  v80 = *v64;
                  if ( v108 == *v64 )
                    goto LABEL_83;
                  ++v81;
                }
                if ( v82 )
                  v64 = v82;
              }
            }
            else if ( v33 - *(_DWORD *)(a1 + 500) - v66 <= v33 >> 3 )
            {
              sub_210FEC0(v34, v33);
              v87 = *(_DWORD *)(a1 + 504);
              if ( !v87 )
              {
LABEL_170:
                ++*(_DWORD *)(a1 + 496);
                BUG();
              }
              v88 = v87 - 1;
              v89 = *(_QWORD *)(a1 + 488);
              v90 = 1;
              LODWORD(v91) = v88 & (((unsigned int)v108 >> 4) ^ ((unsigned int)v108 >> 9));
              v64 = (_QWORD *)(v89 + 72LL * (unsigned int)v91);
              v92 = *v64;
              v66 = *(_DWORD *)(a1 + 496) + 1;
              v93 = 0;
              if ( *v64 != v108 )
              {
                while ( v92 != -8 )
                {
                  if ( !v93 && v92 == -16 )
                    v93 = v64;
                  v91 = v88 & (unsigned int)(v91 + v90);
                  v64 = (_QWORD *)(v89 + 72 * v91);
                  v92 = *v64;
                  if ( v108 == *v64 )
                    goto LABEL_83;
                  ++v90;
                }
                if ( v93 )
                  v64 = v93;
              }
            }
LABEL_83:
            *(_DWORD *)(a1 + 496) = v66;
            if ( *v64 != -8 )
              --*(_DWORD *)(a1 + 500);
            *v64 = v108;
            v64[1] = v64 + 3;
            v64[2] = 0x400000000LL;
            goto LABEL_47;
          }
LABEL_38:
          v39 = v37[1];
          v40 = v39 + 12LL * *((unsigned int *)v37 + 4);
          if ( v39 != v40 )
          {
            v100 = v10;
            v41 = v40;
            do
            {
              while ( 1 )
              {
                v42 = *(unsigned int *)(v39 + 4);
                v43 = *(_DWORD *)v39;
                v44 = 1LL << v42;
                v45 = 8LL * ((unsigned int)v42 >> 6);
                v46 = (__int64 *)(v45 + v110[0]);
                v47 = *(_QWORD *)(v45 + v110[0]);
                v48 = v47 & (1LL << v42);
                if ( *(_BYTE *)(v39 + 8) )
                  break;
                if ( v48 )
                {
                  v58 = (_QWORD *)(*(_QWORD *)(a1 + 168) + 24 * v42);
                  v59 = *(_DWORD *)&v114[4 * v42];
                  if ( v59 != v43 )
                  {
                    v109 = v59 & 0x3F;
                    v60 = (__int64 *)(*v58 + 8LL * (v59 >> 6));
                    v103 = *v60;
                    if ( v59 >> 6 == v43 >> 6 )
                    {
                      *v60 = ((1LL << v43) - (1LL << v109)) | v103;
                      v46 = (__int64 *)(v45 + v110[0]);
                      v47 = *(_QWORD *)(v45 + v110[0]);
                    }
                    else
                    {
                      *v60 = (-1LL << v109) | v103;
                      v61 = (v59 + 63) & 0xFFFFFFC0;
                      for ( j = v61 + 64; v43 >= j; j += 64 )
                      {
                        *(_QWORD *)(*v58 + 8LL * ((j - 64) >> 6)) = -1;
                        v61 = j;
                      }
                      if ( v43 > v61 )
                        *(_QWORD *)(*v58 + 8LL * (v61 >> 6)) |= (1LL << v43) - 1;
                      v46 = (__int64 *)(v45 + v110[0]);
                      v47 = *(_QWORD *)(v45 + v110[0]);
                    }
                  }
                  *v46 = ~v44 & v47;
                }
                v39 += 12;
                *(_QWORD *)(v112[0] + v45) |= v44;
                if ( v41 == v39 )
                  goto LABEL_46;
              }
              if ( !v48 )
              {
                *v46 = v44 | v47;
                *(_QWORD *)(v112[0] + 8LL * ((unsigned int)v42 >> 6)) &= ~v44;
                *(_DWORD *)&v114[4 * v42] = v43;
              }
              v39 += 12;
            }
            while ( v41 != v39 );
LABEL_46:
            v10 = v100;
          }
LABEL_47:
          v49 = v114;
          if ( *(_DWORD *)(a1 + 128) )
          {
            v50 = 0;
            while ( 1 )
            {
              if ( (*(_QWORD *)(v110[0] + 8LL * (v50 >> 6)) & (1LL << v50)) == 0 )
                goto LABEL_49;
              v51 = *(_DWORD *)&v49[4 * v50];
              v52 = (_QWORD *)(*(_QWORD *)(a1 + 168) + 24LL * v50);
              if ( v51 == v21 )
                goto LABEL_49;
              v53 = v51 & 0x3F;
              v54 = (__int64 *)(*v52 + 8LL * (v51 >> 6));
              v55 = *v54;
              if ( v51 >> 6 == v21 >> 6 )
              {
                *v54 = v55 | ((1LL << v21) - (1LL << v53));
                v49 = v114;
LABEL_49:
                if ( *(_DWORD *)(a1 + 128) <= ++v50 )
                  break;
              }
              else
              {
                *v54 = v55 | (-1LL << v53);
                v56 = (v51 + 63) & 0xFFFFFFC0;
                for ( k = v56 + 64; v21 >= k; k += 64 )
                {
                  *(_QWORD *)(*v52 + 8LL * ((k - 64) >> 6)) = -1;
                  v56 = k;
                }
                if ( v21 > v56 )
                  *(_QWORD *)(*v52 + 8LL * (v56 >> 6)) |= (1LL << v21) - 1;
                v49 = v114;
                if ( *(_DWORD *)(a1 + 128) <= ++v50 )
                  break;
              }
            }
          }
          if ( v49 != v116 )
            _libc_free((unsigned __int64)v49);
          v4 += 13;
          _libc_free(v112[0]);
          _libc_free(v110[0]);
          _libc_free((unsigned __int64)v105);
          _libc_free((unsigned __int64)v10);
          _libc_free((unsigned __int64)v107);
          _libc_free((unsigned __int64)v106);
          if ( v4 == v102 )
            return;
          while ( *v4 == -16 || *v4 == -8 )
          {
            v4 += 13;
            if ( v102 == v4 )
              return;
          }
          if ( v102 == v4 )
            return;
          v108 = *v4;
        }
        sub_16BD1C0("Allocation failed", 1u);
LABEL_11:
        memcpy(v106, (const void *)v4[1], v6);
        goto LABEL_12;
      }
    }
  }
}
