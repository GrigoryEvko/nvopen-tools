// Function: sub_1050730
// Address: 0x1050730
//
void __fastcall sub_1050730(__int64 a1)
{
  __int64 v1; // r9
  __int64 *v2; // rax
  __int64 v3; // rcx
  __int64 v4; // rdx
  __int64 *v5; // r13
  __int64 v7; // r8
  __int64 v8; // rcx
  unsigned int v9; // esi
  __int64 v10; // r9
  __int64 v11; // r8
  unsigned int v12; // ebx
  unsigned int v13; // edi
  __int64 *v14; // rax
  __int64 v15; // rcx
  unsigned int v16; // r12d
  int v17; // ebx
  int v18; // ecx
  _BYTE *v19; // r8
  unsigned int v20; // r14d
  __int64 v21; // rax
  int v22; // ecx
  int v23; // eax
  __int64 v24; // rcx
  unsigned __int64 v25; // r8
  int v26; // eax
  unsigned __int64 v27; // r8
  int v28; // r14d
  _BYTE *v29; // rdx
  _DWORD *v30; // rax
  _DWORD *i; // rdx
  int v32; // edx
  unsigned int v33; // ecx
  __int64 v34; // rsi
  __int64 v35; // r8
  __int64 v36; // rdi
  __int64 v37; // rax
  __int64 v38; // rdx
  __int64 v39; // r14
  unsigned int v40; // r8d
  __int64 v41; // rdi
  __int64 v42; // rax
  __int64 *v43; // rbx
  __int64 v44; // r11
  _BYTE *v45; // r8
  unsigned int v46; // edx
  __int64 v47; // rax
  char v48; // cl
  __int64 *v49; // r8
  unsigned int k; // eax
  _QWORD *v51; // r10
  __int64 *v52; // r11
  unsigned int j; // ecx
  unsigned __int64 v54; // r14
  int v55; // r14d
  __int64 *v56; // rdx
  int v57; // eax
  int v58; // r10d
  __int64 *v59; // rdx
  int v60; // eax
  int v61; // eax
  int v62; // r10d
  int v63; // r10d
  __int64 v64; // rcx
  __int64 v65; // rsi
  int v66; // r11d
  __int64 *v67; // rdi
  int v68; // r10d
  int v69; // r10d
  __int64 v70; // rdi
  __int64 v71; // rax
  int v72; // r11d
  __int64 *v73; // r8
  int v74; // r10d
  int v75; // r10d
  int v76; // r8d
  __int64 *v77; // rcx
  unsigned int v78; // ebx
  __int64 v79; // rsi
  int v80; // r10d
  int v81; // r10d
  int v82; // r8d
  unsigned int v83; // ebx
  __int64 v84; // rdi
  __int64 *v85; // rax
  __int64 v86; // [rsp-2A0h] [rbp-2A0h]
  __int64 v87; // [rsp-288h] [rbp-288h]
  __int64 v88; // [rsp-270h] [rbp-270h]
  unsigned __int64 v89; // [rsp-270h] [rbp-270h]
  __int64 v90; // [rsp-240h] [rbp-240h]
  char v91; // [rsp-240h] [rbp-240h]
  _BYTE *v92; // [rsp-238h] [rbp-238h] BYREF
  __int64 v93; // [rsp-230h] [rbp-230h]
  _BYTE v94[32]; // [rsp-228h] [rbp-228h] BYREF
  _BYTE *v95; // [rsp-208h] [rbp-208h] BYREF
  __int64 v96; // [rsp-200h] [rbp-200h]
  _BYTE v97[48]; // [rsp-1F8h] [rbp-1F8h] BYREF
  int v98; // [rsp-1C8h] [rbp-1C8h]
  _BYTE *v99; // [rsp-1B8h] [rbp-1B8h] BYREF
  __int64 v100; // [rsp-1B0h] [rbp-1B0h]
  _BYTE v101[48]; // [rsp-1A8h] [rbp-1A8h] BYREF
  int v102; // [rsp-178h] [rbp-178h]
  __int64 v103; // [rsp-168h] [rbp-168h]
  _QWORD v104[2]; // [rsp-160h] [rbp-160h] BYREF
  _DWORD v105[14]; // [rsp-150h] [rbp-150h] BYREF
  _QWORD v106[2]; // [rsp-118h] [rbp-118h] BYREF
  _DWORD v107[14]; // [rsp-108h] [rbp-108h] BYREF
  _QWORD v108[2]; // [rsp-D0h] [rbp-D0h] BYREF
  _DWORD v109[14]; // [rsp-C0h] [rbp-C0h] BYREF
  _QWORD v110[2]; // [rsp-88h] [rbp-88h] BYREF
  _DWORD v111[30]; // [rsp-78h] [rbp-78h] BYREF

  v1 = *(unsigned int *)(a1 + 32);
  if ( (_DWORD)v1 )
  {
    v2 = *(__int64 **)(a1 + 24);
    v3 = (__int64)&v2[37 * *(unsigned int *)(a1 + 40)];
    v87 = v3;
    if ( v2 != (__int64 *)v3 )
    {
      while ( 1 )
      {
        v4 = *v2;
        v5 = v2;
        if ( *v2 != -8192 && v4 != -4096 )
          break;
        v2 += 37;
        if ( (__int64 *)v3 == v2 )
          return;
      }
      if ( (__int64 *)v3 != v2 )
      {
        v86 = a1 + 576;
        while ( 1 )
        {
          v103 = *v5;
          v104[0] = v105;
          v104[1] = 0x600000000LL;
          v7 = *((unsigned int *)v5 + 4);
          if ( (_DWORD)v7 )
            sub_104D010((__int64)v104, (__int64)(v5 + 1), v4, v3, v7, v1);
          v105[12] = *((_DWORD *)v5 + 18);
          v106[0] = v107;
          v106[1] = 0x600000000LL;
          if ( *((_DWORD *)v5 + 22) )
            sub_104D010((__int64)v106, (__int64)(v5 + 10), v4, v3, v7, v1);
          v107[12] = *((_DWORD *)v5 + 36);
          v108[0] = v109;
          v108[1] = 0x600000000LL;
          if ( *((_DWORD *)v5 + 40) )
            sub_104D010((__int64)v108, (__int64)(v5 + 19), v4, v3, v7, v1);
          v109[12] = *((_DWORD *)v5 + 54);
          v110[0] = v111;
          v110[1] = 0x600000000LL;
          v8 = *((unsigned int *)v5 + 58);
          if ( (_DWORD)v8 )
            sub_104D010((__int64)v110, (__int64)(v5 + 28), v4, v8, v7, v1);
          v9 = *(_DWORD *)(a1 + 600);
          v111[12] = *((_DWORD *)v5 + 72);
          v90 = v103;
          if ( v9 )
          {
            v10 = v9 - 1;
            v11 = *(_QWORD *)(a1 + 584);
            v12 = ((unsigned int)v103 >> 9) ^ ((unsigned int)v103 >> 4);
            v13 = v10 & v12;
            v14 = (__int64 *)(v11 + 16LL * ((unsigned int)v10 & v12));
            v15 = *v14;
            if ( v103 == *v14 )
            {
LABEL_19:
              v16 = *((_DWORD *)v14 + 3);
              v17 = *((_DWORD *)v14 + 2);
              goto LABEL_20;
            }
            v58 = 1;
            v59 = 0;
            while ( v15 != -4096 )
            {
              if ( v15 == -8192 && !v59 )
                v59 = v14;
              v13 = v10 & (v58 + v13);
              v14 = (__int64 *)(v11 + 16LL * v13);
              v15 = *v14;
              if ( v103 == *v14 )
                goto LABEL_19;
              ++v58;
            }
            if ( !v59 )
              v59 = v14;
            v60 = *(_DWORD *)(a1 + 592);
            ++*(_QWORD *)(a1 + 576);
            v61 = v60 + 1;
            if ( 4 * v61 < 3 * v9 )
            {
              if ( v9 - *(_DWORD *)(a1 + 596) - v61 <= v9 >> 3 )
              {
                sub_104FE00(v86, v9);
                v74 = *(_DWORD *)(a1 + 600);
                if ( !v74 )
                {
LABEL_178:
                  ++*(_DWORD *)(a1 + 592);
                  BUG();
                }
                v75 = v74 - 1;
                v10 = *(_QWORD *)(a1 + 584);
                v76 = 1;
                v77 = 0;
                v78 = v75 & v12;
                v61 = *(_DWORD *)(a1 + 592) + 1;
                v59 = (__int64 *)(v10 + 16LL * v78);
                v79 = *v59;
                if ( v90 != *v59 )
                {
                  while ( v79 != -4096 )
                  {
                    if ( !v77 && v79 == -8192 )
                      v77 = v59;
                    v78 = v75 & (v76 + v78);
                    v59 = (__int64 *)(v10 + 16LL * v78);
                    v79 = *v59;
                    if ( v90 == *v59 )
                      goto LABEL_117;
                    ++v76;
                  }
                  if ( v77 )
                    v59 = v77;
                }
              }
              goto LABEL_117;
            }
          }
          else
          {
            ++*(_QWORD *)(a1 + 576);
          }
          sub_104FE00(v86, 2 * v9);
          v62 = *(_DWORD *)(a1 + 600);
          if ( !v62 )
            goto LABEL_178;
          v63 = v62 - 1;
          v10 = *(_QWORD *)(a1 + 584);
          v64 = v63 & (((unsigned int)v90 >> 9) ^ ((unsigned int)v90 >> 4));
          v61 = *(_DWORD *)(a1 + 592) + 1;
          v59 = (__int64 *)(v10 + 16 * v64);
          v65 = *v59;
          if ( v90 != *v59 )
          {
            v66 = 1;
            v67 = 0;
            while ( v65 != -4096 )
            {
              if ( v65 == -8192 && !v67 )
                v67 = v59;
              LODWORD(v64) = v63 & (v66 + v64);
              v59 = (__int64 *)(v10 + 16LL * (unsigned int)v64);
              v65 = *v59;
              if ( v90 == *v59 )
                goto LABEL_117;
              ++v66;
            }
            if ( v67 )
              v59 = v67;
          }
LABEL_117:
          *(_DWORD *)(a1 + 592) = v61;
          if ( *v59 != -4096 )
            --*(_DWORD *)(a1 + 596);
          v59[1] = 0;
          v17 = 0;
          v16 = 0;
          *v59 = v90;
LABEL_20:
          v18 = *(_DWORD *)(a1 + 624);
          v19 = v97;
          v95 = v97;
          v20 = (unsigned int)(v18 + 63) >> 6;
          v96 = 0x600000000LL;
          v99 = v101;
          v100 = 0x600000000LL;
          v102 = 0;
          v98 = v18;
          if ( v20 )
          {
            v21 = v20;
            if ( v20 > 6 )
            {
              sub_C8D5F0((__int64)&v95, v97, v20, 8u, (__int64)v97, v10);
              v21 = v20;
              v19 = &v95[8 * (unsigned int)v96];
            }
            memset(v19, 0, 8 * v21);
            LODWORD(v96) = v20 + v96;
            LOBYTE(v18) = v98;
          }
          v22 = v18 & 0x3F;
          if ( v22 )
            *(_QWORD *)&v95[8 * (unsigned int)v96 - 8] &= ~(-1LL << v22);
          v23 = *(_DWORD *)(a1 + 624);
          if ( (v102 & 0x3F) != 0 )
            *(_QWORD *)&v99[8 * (unsigned int)v100 - 8] &= ~(-1LL << (v102 & 0x3F));
          v24 = (unsigned int)v100;
          v102 = v23;
          v25 = (unsigned int)(v23 + 63) >> 6;
          if ( v25 != (unsigned int)v100 )
          {
            if ( v25 >= (unsigned int)v100 )
            {
              v54 = v25 - (unsigned int)v100;
              if ( v25 > HIDWORD(v100) )
              {
                sub_C8D5F0((__int64)&v99, v101, v25, 8u, v25, v10);
                v24 = (unsigned int)v100;
              }
              if ( 8 * v54 )
              {
                memset(&v99[8 * v24], 0, 8 * v54);
                LODWORD(v24) = v100;
              }
              LOBYTE(v23) = v102;
              LODWORD(v100) = v54 + v24;
            }
            else
            {
              LODWORD(v100) = (unsigned int)(v23 + 63) >> 6;
            }
          }
          v26 = v23 & 0x3F;
          if ( v26 )
            *(_QWORD *)&v99[8 * (unsigned int)v100 - 8] &= ~(-1LL << v26);
          v27 = *(unsigned int *)(a1 + 624);
          v92 = v94;
          v28 = v27;
          v93 = 0x800000000LL;
          if ( v27 )
          {
            v29 = v94;
            v30 = v94;
            if ( v27 > 8 )
            {
              v89 = v27;
              sub_C8D5F0((__int64)&v92, v94, v27, 4u, v27, v10);
              v29 = v92;
              v27 = v89;
              v30 = &v92[4 * (unsigned int)v93];
            }
            for ( i = &v29[4 * v27]; i != v30; ++v30 )
            {
              if ( v30 )
                *v30 = 0;
            }
            v32 = *(_DWORD *)(a1 + 624);
            LODWORD(v93) = v28;
            if ( v32 )
            {
              v33 = 0;
              do
              {
                if ( (*(_QWORD *)(v108[0] + 8LL * (v33 >> 6)) & (1LL << v33)) != 0 )
                {
                  *(_QWORD *)&v95[8 * (v33 >> 6)] |= 1LL << v33;
                  *(_DWORD *)&v92[4 * v33] = v17;
                }
                ++v33;
              }
              while ( *(_DWORD *)(a1 + 624) > v33 );
            }
          }
          v34 = *(unsigned int *)(a1 + 1352);
          if ( !(_DWORD)v34 )
          {
            ++*(_QWORD *)(a1 + 1328);
            goto LABEL_129;
          }
          v1 = (unsigned int)(v34 - 1);
          v35 = *(_QWORD *)(a1 + 1336);
          v36 = (unsigned int)v1 & (((unsigned int)v90 >> 9) ^ ((unsigned int)v90 >> 4));
          v37 = v35 + 72 * v36;
          v3 = *(_QWORD *)v37;
          if ( v90 != *(_QWORD *)v37 )
          {
            v55 = 1;
            v56 = 0;
            while ( v3 != -4096 )
            {
              if ( !v56 && v3 == -8192 )
                v56 = (__int64 *)v37;
              v36 = (unsigned int)v1 & ((_DWORD)v36 + v55);
              v37 = v35 + 72 * v36;
              v3 = *(_QWORD *)v37;
              if ( v90 == *(_QWORD *)v37 )
                goto LABEL_47;
              ++v55;
            }
            if ( !v56 )
              v56 = (__int64 *)v37;
            v57 = *(_DWORD *)(a1 + 1344);
            ++*(_QWORD *)(a1 + 1328);
            v3 = (unsigned int)(v57 + 1);
            if ( 4 * (int)v3 >= (unsigned int)(3 * v34) )
            {
LABEL_129:
              sub_104FFE0(a1 + 1328, 2 * v34);
              v68 = *(_DWORD *)(a1 + 1352);
              if ( !v68 )
                goto LABEL_179;
              v69 = v68 - 1;
              v1 = *(_QWORD *)(a1 + 1336);
              v34 = *(unsigned int *)(a1 + 1344);
              v3 = (unsigned int)(v34 + 1);
              v70 = v69 & (((unsigned int)v90 >> 9) ^ ((unsigned int)v90 >> 4));
              v56 = (__int64 *)(v1 + 72 * v70);
              v71 = *v56;
              if ( v90 != *v56 )
              {
                v72 = 1;
                v73 = 0;
                while ( v71 != -4096 )
                {
                  if ( !v73 && v71 == -8192 )
                    v73 = v56;
                  v34 = (unsigned int)(v72 + 1);
                  LODWORD(v70) = v69 & (v72 + v70);
                  v56 = (__int64 *)(v1 + 72LL * (unsigned int)v70);
                  v71 = *v56;
                  if ( v90 == *v56 )
                    goto LABEL_108;
                  ++v72;
                }
                if ( v73 )
                  v56 = v73;
              }
            }
            else if ( (int)v34 - *(_DWORD *)(a1 + 1348) - (int)v3 <= (unsigned int)v34 >> 3 )
            {
              sub_104FFE0(a1 + 1328, v34);
              v80 = *(_DWORD *)(a1 + 1352);
              if ( !v80 )
              {
LABEL_179:
                ++*(_DWORD *)(a1 + 1344);
                BUG();
              }
              v81 = v80 - 1;
              v1 = *(_QWORD *)(a1 + 1336);
              v82 = 1;
              v83 = v81 & (((unsigned int)v90 >> 9) ^ ((unsigned int)v90 >> 4));
              v56 = (__int64 *)(v1 + 72LL * v83);
              v84 = *v56;
              v3 = (unsigned int)(*(_DWORD *)(a1 + 1344) + 1);
              v85 = 0;
              if ( v90 != *v56 )
              {
                while ( v84 != -4096 )
                {
                  if ( !v85 && v84 == -8192 )
                    v85 = v56;
                  v34 = (unsigned int)(v82 + 1);
                  v83 = v81 & (v82 + v83);
                  v56 = (__int64 *)(v1 + 72LL * v83);
                  v84 = *v56;
                  if ( v90 == *v56 )
                    goto LABEL_108;
                  ++v82;
                }
                if ( v85 )
                  v56 = v85;
              }
            }
LABEL_108:
            *(_DWORD *)(a1 + 1344) = v3;
            if ( *v56 != -4096 )
              --*(_DWORD *)(a1 + 1348);
            *v56 = v90;
            v56[1] = (__int64)(v56 + 3);
            v56[2] = 0x400000000LL;
            goto LABEL_55;
          }
LABEL_47:
          v38 = *(_QWORD *)(v37 + 8);
          v39 = v38 + 12LL * *(unsigned int *)(v37 + 16);
          if ( v39 != v38 )
          {
            v1 = 1;
            do
            {
              while ( 1 )
              {
                v3 = *(unsigned int *)(v38 + 4);
                v40 = *(_DWORD *)v38;
                v41 = 1LL << v3;
                v42 = 8LL * ((unsigned int)v3 >> 6);
                v43 = (__int64 *)&v95[v42];
                v44 = *(_QWORD *)&v95[v42];
                v34 = v44 & (1LL << v3);
                if ( *(_BYTE *)(v38 + 8) )
                  break;
                if ( v34 )
                {
                  v51 = (_QWORD *)(*(_QWORD *)(a1 + 664) + 72 * v3);
                  v34 = *(unsigned int *)&v92[4 * v3];
                  if ( (_DWORD)v34 != v40 )
                  {
                    v91 = v34 & 0x3F;
                    v52 = (__int64 *)(*v51 + 8LL * ((unsigned int)v34 >> 6));
                    v88 = *v52;
                    if ( (unsigned int)v34 >> 6 == v40 >> 6 )
                    {
                      v34 = 1LL << v91;
                      *v52 = ((1LL << v40) - (1LL << v91)) | v88;
                      v43 = (__int64 *)&v95[v42];
                      v44 = *(_QWORD *)&v95[v42];
                    }
                    else
                    {
                      *v52 = (-1LL << v91) | v88;
                      v34 = ((unsigned int)((v34 - (unsigned __int64)(v34 != 0)) >> 6) + (v34 != 0)) << 6;
                      for ( j = v34 + 64; v40 >= j; j += 64 )
                      {
                        *(_QWORD *)(*v51 + 8LL * ((j - 64) >> 6)) = -1;
                        v34 = j;
                      }
                      if ( v40 > (unsigned int)v34 )
                      {
                        v34 = (unsigned int)v34 >> 6;
                        *(_QWORD *)(*v51 + 8 * v34) |= (1LL << v40) - 1;
                      }
                      v43 = (__int64 *)&v95[v42];
                      v44 = *(_QWORD *)&v95[v42];
                    }
                  }
                  v3 = ~v41;
                  *v43 = ~v41 & v44;
                }
                v38 += 12;
                *(_QWORD *)&v99[v42] |= v41;
                if ( v39 == v38 )
                  goto LABEL_55;
              }
              if ( !v34 )
              {
                *v43 = v41 | v44;
                *(_QWORD *)&v99[8 * ((unsigned int)v3 >> 6)] &= ~v41;
                *(_DWORD *)&v92[4 * v3] = v40;
              }
              v38 += 12;
            }
            while ( v39 != v38 );
          }
LABEL_55:
          v45 = v92;
          if ( *(_DWORD *)(a1 + 624) )
          {
            v46 = 0;
            while ( 1 )
            {
              v3 = (__int64)v95;
              v34 = v46 >> 6;
              if ( (*(_QWORD *)&v95[8 * v34] & (1LL << v46)) == 0 )
                goto LABEL_57;
              v3 = *(_QWORD *)(a1 + 664);
              v47 = *(unsigned int *)&v45[4 * v46];
              v34 = v3 + 72LL * v46;
              if ( (_DWORD)v47 == v16 )
                goto LABEL_57;
              v48 = v47 & 0x3F;
              v49 = (__int64 *)(*(_QWORD *)v34 + 8LL * ((unsigned int)v47 >> 6));
              v1 = *v49;
              if ( (unsigned int)v47 >> 6 == v16 >> 6 )
              {
                v3 = v1 | ((1LL << v16) - (1LL << v48));
                *v49 = v3;
                v45 = v92;
LABEL_57:
                if ( *(_DWORD *)(a1 + 624) <= ++v46 )
                  break;
              }
              else
              {
                *v49 = v1 | (-1LL << v48);
                v3 = ((v47 != 0) + (unsigned int)((v47 - (unsigned __int64)(v47 != 0)) >> 6)) << 6;
                for ( k = v3 + 64; k <= v16; k += 64 )
                {
                  *(_QWORD *)(*(_QWORD *)v34 + 8LL * ((k - 64) >> 6)) = -1;
                  v3 = k;
                }
                if ( v16 > (unsigned int)v3 )
                {
                  v3 = (unsigned int)v3 >> 6;
                  *(_QWORD *)(*(_QWORD *)v34 + 8 * v3) |= (1LL << v16) - 1;
                }
                v45 = v92;
                if ( *(_DWORD *)(a1 + 624) <= ++v46 )
                  break;
              }
            }
          }
          if ( v45 != v94 )
            _libc_free(v45, v34);
          if ( v99 != v101 )
            _libc_free(v99, v34);
          if ( v95 != v97 )
            _libc_free(v95, v34);
          if ( (_DWORD *)v110[0] != v111 )
            _libc_free(v110[0], v34);
          if ( (_DWORD *)v108[0] != v109 )
            _libc_free(v108[0], v34);
          if ( (_DWORD *)v106[0] != v107 )
            _libc_free(v106[0], v34);
          if ( (_DWORD *)v104[0] != v105 )
            _libc_free(v104[0], v34);
          v4 = v87;
          v5 += 37;
          if ( v5 != (__int64 *)v87 )
          {
            while ( *v5 == -8192 || *v5 == -4096 )
            {
              v5 += 37;
              if ( (__int64 *)v87 == v5 )
                return;
            }
            if ( (__int64 *)v87 != v5 )
              continue;
          }
          return;
        }
      }
    }
  }
}
