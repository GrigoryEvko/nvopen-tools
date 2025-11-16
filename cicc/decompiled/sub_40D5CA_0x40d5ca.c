// Function: sub_40D5CA
// Address: 0x40d5ca
//
unsigned __int64 __fastcall sub_40D5CA(__int64 a1, unsigned __int64 a2, char *a3, __int64 a4)
{
  unsigned __int64 v4; // r15
  char v8; // al
  char *v9; // rdx
  char v10; // r11
  char *v11; // rdi
  char v12; // al
  int v13; // r12d
  unsigned int v14; // edx
  int *v15; // rax
  int v16; // eax
  char *v17; // rax
  int v18; // edx
  char v19; // cl
  int v20; // eax
  unsigned int v21; // ecx
  int *v22; // rdx
  char *v23; // rax
  char v24; // si
  unsigned __int8 v25; // cl
  char *v26; // r10
  unsigned int v27; // eax
  unsigned __int64 v28; // rax
  int *v29; // rdx
  __int64 v30; // rcx
  __int64 *v31; // rdx
  int v32; // edi
  __int64 v33; // r9
  unsigned __int64 v34; // r8
  _BYTE *v35; // rsi
  unsigned __int64 v36; // rdx
  char v37; // dl
  unsigned __int64 v38; // rax
  unsigned __int64 v39; // rdx
  unsigned __int64 v40; // rcx
  unsigned __int64 v41; // rdi
  unsigned __int64 v42; // rcx
  unsigned __int64 i; // rcx
  unsigned int *v44; // rdx
  unsigned __int64 v45; // rax
  unsigned __int64 *v46; // rdx
  __int64 v47; // rdx
  char v48; // cl
  unsigned __int64 v49; // r9
  _BYTE *v50; // rsi
  char v51; // cl
  unsigned __int64 v52; // rax
  unsigned __int64 v53; // rcx
  unsigned __int64 v54; // rdi
  unsigned __int64 v55; // rcx
  unsigned __int64 j; // rcx
  unsigned int *v57; // rdx
  unsigned __int64 v58; // rcx
  unsigned __int64 *v59; // rdx
  __int64 v60; // rdi
  _BYTE *v61; // rsi
  unsigned __int64 v62; // rdx
  unsigned __int64 v63; // rax
  unsigned __int64 v64; // rcx
  unsigned __int64 v65; // rdi
  unsigned __int64 v66; // rcx
  unsigned __int64 n; // rcx
  unsigned int v68; // eax
  unsigned int *v69; // rdx
  unsigned __int64 v70; // rdi
  unsigned __int64 *v71; // rdx
  _WORD *v72; // rax
  unsigned __int64 v73; // rdx
  const void *v74; // rsi
  unsigned __int64 v75; // rax
  unsigned __int64 v76; // rcx
  unsigned __int64 v77; // rax
  unsigned __int64 v78; // rcx
  unsigned __int64 ii; // rcx
  int *v80; // rdx
  int v81; // edi
  unsigned __int64 v82; // rax
  unsigned __int64 v83; // rcx
  unsigned __int64 v84; // rcx
  __int64 v85; // r15
  const char **v86; // rcx
  const char *v87; // rsi
  size_t v88; // rax
  unsigned __int64 v89; // rcx
  unsigned __int64 v90; // rdi
  unsigned __int64 v91; // rcx
  unsigned __int64 m; // rcx
  unsigned __int64 *v93; // rdi
  _WORD *v94; // rax
  const void *v95; // rsi
  unsigned __int64 v96; // rax
  unsigned __int64 v97; // rcx
  unsigned __int64 v98; // rax
  unsigned __int64 v99; // rcx
  unsigned __int64 k; // rcx
  char v102; // [rsp+8h] [rbp-B8h]
  char *v103; // [rsp+8h] [rbp-B8h]
  char *v104; // [rsp+10h] [rbp-B0h]
  char v105; // [rsp+10h] [rbp-B0h]
  char v106; // [rsp+18h] [rbp-A8h]
  char v107; // [rsp+18h] [rbp-A8h]
  char *v108; // [rsp+18h] [rbp-A8h]
  char v109; // [rsp+20h] [rbp-A0h]
  char v110; // [rsp+20h] [rbp-A0h]
  char v111; // [rsp+2Eh] [rbp-92h]
  char v112; // [rsp+2Eh] [rbp-92h]
  char v113; // [rsp+2Fh] [rbp-91h]
  char *v114; // [rsp+38h] [rbp-88h] BYREF
  unsigned __int64 v115; // [rsp+40h] [rbp-80h] BYREF
  _BYTE v116[64]; // [rsp+4Dh] [rbp-73h] BYREF
  char v117; // [rsp+8Dh] [rbp-33h]

  v4 = 0;
  v114 = a3;
  while ( 1 )
  {
    v8 = *v114;
    if ( !*v114 )
      break;
    v9 = v114 + 1;
    if ( v8 == 37 )
    {
      ++v114;
      v10 = 0;
      v111 = 0;
      v106 = 0;
      v109 = 0;
      while ( 1 )
      {
        v11 = v114;
        v12 = *v114;
        if ( *v114 == 43 )
        {
          v111 = 1;
          goto LABEL_8;
        }
        if ( *v114 <= 43 )
          break;
        if ( v12 != 45 )
        {
          if ( v12 == 48 )
          {
            v113 = 1;
          }
          else
          {
            if ( v12 <= 48 )
              goto LABEL_19;
            v113 = 0;
            v13 = -1;
            if ( (unsigned __int8)(v12 - 49) > 8u )
              goto LABEL_32;
          }
          v102 = v10;
          v104 = v114;
          *__errno_location() = 0;
          v16 = sub_130AAB0(v104, &v114, 10);
          v10 = v102;
          v13 = v16;
          goto LABEL_32;
        }
        v10 = 1;
LABEL_8:
        ++v114;
      }
      if ( v12 == 32 )
      {
        v106 = 1;
        goto LABEL_8;
      }
      if ( v12 == 35 )
      {
        v109 = 1;
        goto LABEL_8;
      }
      if ( v12 != 42 )
      {
LABEL_19:
        v113 = 0;
        v13 = -1;
        goto LABEL_32;
      }
      v14 = *(_DWORD *)a4;
      if ( *(_DWORD *)a4 > 0x2Fu )
      {
        v15 = *(int **)(a4 + 8);
        *(_QWORD *)(a4 + 8) = v15 + 2;
      }
      else
      {
        v15 = (int *)(*(_QWORD *)(a4 + 16) + v14);
        *(_DWORD *)a4 = v14 + 8;
      }
      v13 = *v15;
      v113 = 0;
      v114 = v11 + 1;
      if ( v13 < 0 )
      {
        v13 = -v13;
        v10 = 1;
      }
LABEL_32:
      v17 = v114;
      v18 = -1;
      if ( *v114 == 46 )
      {
        ++v114;
        v19 = v17[1];
        v103 = v17 + 1;
        if ( v19 == 42 )
        {
          v21 = *(_DWORD *)a4;
          if ( *(_DWORD *)a4 > 0x2Fu )
          {
            v22 = *(int **)(a4 + 8);
            *(_QWORD *)(a4 + 8) = v22 + 2;
          }
          else
          {
            v22 = (int *)(*(_QWORD *)(a4 + 16) + v21);
            *(_DWORD *)a4 = v21 + 8;
          }
          v18 = *v22;
          v114 = v17 + 2;
        }
        else if ( (unsigned __int8)(v19 - 48) <= 9u )
        {
          v105 = v10;
          *__errno_location() = 0;
          v20 = sub_130AAB0(v103, &v114, 10);
          v10 = v105;
          v18 = v20;
        }
      }
      v23 = v114;
      v24 = *v114;
      v25 = *v114 - 106;
      if ( v25 > 0x10u )
      {
        v24 = 63;
      }
      else if ( ((1LL << v25) & 0x10481) != 0 )
      {
        ++v114;
      }
      else
      {
        v24 = 63;
        if ( ((1LL << v25) & 4) != 0 )
        {
          v24 = 108;
          ++v114;
          if ( v23[1] == 108 )
          {
            v24 = 113;
            v114 = v23 + 2;
          }
        }
      }
      v26 = v114;
      if ( *v114 > 98 )
      {
        v27 = *(_DWORD *)a4;
        switch ( *(_DWORD *)a4 )
        {
          case 0:
            if ( v27 > 0x2F )
            {
              v80 = *(int **)(a4 + 8);
              *(_QWORD *)(a4 + 8) = v80 + 2;
            }
            else
            {
              v80 = (int *)(*(_QWORD *)(a4 + 16) + v27);
              *(_DWORD *)a4 = v27 + 8;
            }
            v81 = *v80;
            v39 = 0;
            if ( (unsigned int)(v13 + 1) > 2 )
              v39 = v13 - 1LL;
            if ( !v10 && v39 )
            {
              v82 = v4;
              do
              {
                v83 = v82 + 1;
                if ( v113 )
                {
                  if ( a2 > v82 )
                    *(_BYTE *)(a1 + v82) = 48;
                }
                else if ( a2 > v82 )
                {
                  *(_BYTE *)(a1 + v82) = 32;
                }
                ++v82;
              }
              while ( v39 > v83 - v4 );
              v4 += v39;
            }
            if ( v4 < a2 )
              *(_BYTE *)(a1 + v4) = v81;
            v28 = v4 + 1;
            if ( !v39 || !v10 )
              goto LABEL_259;
            v84 = v4 + 1;
            v85 = ~v4;
            do
            {
              if ( a2 > v84 )
                *(_BYTE *)(a1 + v84) = 32;
              ++v84;
            }
            while ( v39 > v85 + v84 );
            goto LABEL_258;
          case 1:
          case 6:
            if ( v24 == 63 )
            {
              if ( v27 > 0x2F )
              {
                v29 = *(int **)(a4 + 8);
                *(_QWORD *)(a4 + 8) = v29 + 2;
              }
              else
              {
                v29 = (int *)(*(_QWORD *)(a4 + 16) + v27);
                *(_DWORD *)a4 = v27 + 8;
              }
              v30 = *v29;
            }
            else
            {
              if ( v27 > 0x2F )
              {
                v31 = *(__int64 **)(a4 + 8);
                *(_QWORD *)(a4 + 8) = v31 + 1;
              }
              else
              {
                v31 = (__int64 *)(*(_QWORD *)(a4 + 16) + v27);
                *(_DWORD *)a4 = v27 + 8;
              }
              v30 = *v31;
            }
            v32 = 43;
            if ( !v111 )
              v32 = v106 == 0 ? 45 : 32;
            v117 = 0;
            LODWORD(v33) = 64;
            v34 = abs64(v30);
            do
            {
              v33 = (unsigned int)(v33 - 1);
              v35 = &v116[v33];
              v116[v33] = a0123456789[v34 % 0xA + 1];
              v36 = v34;
              v34 /= 0xAu;
            }
            while ( v36 > 9 );
            if ( v30 < 0 )
            {
              v37 = 45;
            }
            else
            {
              v37 = v32;
              if ( v32 == 45 )
              {
                v115 = (unsigned int)(64 - v33);
                goto LABEL_75;
              }
              if ( (_BYTE)v32 != 32 )
                v37 = 43;
            }
            *--v35 = v37;
            v115 = (unsigned int)(64 - v33) + 1LL;
LABEL_75:
            v38 = v115;
            v39 = 0;
            if ( v13 != -1 && v13 > v115 )
              v39 = v13 - v115;
            if ( !v10 && v39 )
            {
              v40 = v4;
              do
              {
                v41 = v40 + 1;
                if ( v113 )
                {
                  if ( a2 > v40 )
                    *(_BYTE *)(a1 + v40) = 48;
                }
                else if ( a2 > v40 )
                {
                  *(_BYTE *)(a1 + v40) = 32;
                }
                ++v40;
              }
              while ( v39 > v41 - v4 );
              v4 += v39;
            }
            if ( v4 < a2 )
            {
              v42 = a2 - v4;
              if ( a2 - v4 > v38 )
                v42 = v38;
              qmemcpy((void *)(a1 + v4), v35, v42);
            }
            v28 = v4 + v38;
            if ( v39 && v10 )
            {
              for ( i = 0; i < v39; ++i )
              {
                if ( a2 > v28 + i )
                  *(_BYTE *)(a1 + v28 + i) = 32;
              }
LABEL_258:
              v28 += v39;
            }
LABEL_259:
            v114 = v26 + 1;
            break;
          case 2:
          case 3:
          case 4:
          case 5:
          case 7:
          case 8:
          case 9:
          case 0xA:
          case 0xB:
          case 0xE:
          case 0xF:
          case 0x11:
          case 0x13:
          case 0x14:
          case 0x15:
            goto LABEL_173;
          case 0xC:
            if ( ((unsigned __int8)v24 | 0x80) == 0xBF )
            {
              if ( v27 > 0x2F )
              {
                v44 = *(unsigned int **)(a4 + 8);
                *(_QWORD *)(a4 + 8) = v44 + 2;
              }
              else
              {
                v44 = (unsigned int *)(*(_QWORD *)(a4 + 16) + v27);
                *(_DWORD *)a4 = v27 + 8;
              }
              v45 = *v44;
            }
            else
            {
              if ( v27 > 0x2F )
              {
                v46 = *(unsigned __int64 **)(a4 + 8);
                *(_QWORD *)(a4 + 8) = v46 + 1;
              }
              else
              {
                v46 = (unsigned __int64 *)(*(_QWORD *)(a4 + 16) + v27);
                *(_DWORD *)a4 = v27 + 8;
              }
              v45 = *v46;
            }
            v117 = 0;
            LODWORD(v47) = 64;
            do
            {
              v48 = v45;
              v49 = v45;
              v45 >>= 3;
              v47 = (unsigned int)(v47 - 1);
              v50 = &v116[v47];
              v51 = a0123456789abcd_9[v48 & 7];
              v116[v47] = v51;
            }
            while ( v49 > 7 );
            if ( v51 != 48 && v109 )
            {
              *--v50 = 48;
              v115 = (unsigned int)(64 - v47) + 1LL;
            }
            else
            {
              v115 = (unsigned int)(64 - v47);
            }
            v52 = v115;
            v39 = 0;
            if ( v13 != -1 && v13 > v115 )
              v39 = v13 - v115;
            if ( !v10 && v39 )
            {
              v53 = v4;
              do
              {
                v54 = v53 + 1;
                if ( v113 )
                {
                  if ( a2 > v53 )
                    *(_BYTE *)(a1 + v53) = 48;
                }
                else if ( a2 > v53 )
                {
                  *(_BYTE *)(a1 + v53) = 32;
                }
                ++v53;
              }
              while ( v39 > v54 - v4 );
              v4 += v39;
            }
            if ( v4 < a2 )
            {
              v55 = a2 - v4;
              if ( a2 - v4 > v52 )
                v55 = v52;
              qmemcpy((void *)(a1 + v4), v50, v55);
            }
            v28 = v4 + v52;
            if ( !v39 || !v10 )
              goto LABEL_259;
            for ( j = 0; j < v39; ++j )
            {
              if ( a2 > v28 + j )
                *(_BYTE *)(a1 + v28 + j) = 32;
            }
            goto LABEL_258;
          case 0xD:
            if ( v27 > 0x2F )
            {
              v93 = *(unsigned __int64 **)(a4 + 8);
              *(_QWORD *)(a4 + 8) = v93 + 1;
            }
            else
            {
              v93 = (unsigned __int64 *)(*(_QWORD *)(a4 + 16) + v27);
              *(_DWORD *)a4 = v27 + 8;
            }
            v110 = v10;
            v94 = sub_40D550(*v93, 1, 0, (__int64)v116, &v115);
            v73 = 0;
            v95 = v94;
            if ( v13 != -1 && v13 > v115 )
              v73 = v13 - v115;
            if ( !v110 && v73 )
            {
              v96 = v4;
              do
              {
                v97 = v96 + 1;
                if ( v113 )
                {
                  if ( a2 > v96 )
                    *(_BYTE *)(a1 + v96) = 48;
                }
                else if ( a2 > v96 )
                {
                  *(_BYTE *)(a1 + v96) = 32;
                }
                ++v96;
              }
              while ( v73 > v97 - v4 );
              v4 += v73;
            }
            v98 = v115;
            if ( v4 < a2 )
            {
              v99 = a2 - v4;
              if ( a2 - v4 > v115 )
                v99 = v115;
              qmemcpy((void *)(a1 + v4), v95, v99);
            }
            v28 = v4 + v98;
            if ( !v73 || !v110 )
              goto LABEL_287;
            for ( k = 0; k < v73; ++k )
            {
              if ( a2 > v28 + k )
                *(_BYTE *)(a1 + v28 + k) = 32;
            }
            goto LABEL_286;
          case 0x10:
            if ( v27 > 0x2F )
            {
              v86 = *(const char ***)(a4 + 8);
              *(_QWORD *)(a4 + 8) = v86 + 1;
            }
            else
            {
              v86 = (const char **)(*(_QWORD *)(a4 + 16) + v27);
              *(_DWORD *)a4 = v27 + 8;
            }
            v87 = *v86;
            v88 = v18;
            if ( v18 < 0 )
            {
              v112 = v10;
              v108 = v26;
              v88 = strlen(*v86);
              v10 = v112;
              v26 = v108;
            }
            v115 = v88;
            v39 = 0;
            if ( v13 != -1 && v13 > v88 )
              v39 = v13 - v88;
            if ( !v10 && v39 )
            {
              v89 = v4;
              do
              {
                v90 = v89 + 1;
                if ( v113 )
                {
                  if ( v89 < a2 )
                    *(_BYTE *)(a1 + v89) = 48;
                }
                else if ( v89 < a2 )
                {
                  *(_BYTE *)(a1 + v89) = 32;
                }
                ++v89;
              }
              while ( v90 - v4 < v39 );
              v4 += v39;
            }
            if ( v4 < a2 )
            {
              v91 = a2 - v4;
              if ( a2 - v4 > v88 )
                v91 = v88;
              qmemcpy((void *)(a1 + v4), v87, v91);
            }
            v28 = v4 + v88;
            if ( !v39 || !v10 )
              goto LABEL_259;
            for ( m = 0; m < v39; ++m )
            {
              if ( v28 + m < a2 )
                *(_BYTE *)(a1 + v28 + m) = 32;
            }
            goto LABEL_258;
          case 0x12:
            if ( ((unsigned __int8)v24 | 0x80) == 0xBF )
            {
              if ( v27 > 0x2F )
              {
                v57 = *(unsigned int **)(a4 + 8);
                *(_QWORD *)(a4 + 8) = v57 + 2;
              }
              else
              {
                v57 = (unsigned int *)(*(_QWORD *)(a4 + 16) + v27);
                *(_DWORD *)a4 = v27 + 8;
              }
              v58 = *v57;
            }
            else
            {
              if ( v27 > 0x2F )
              {
                v59 = *(unsigned __int64 **)(a4 + 8);
                *(_QWORD *)(a4 + 8) = v59 + 1;
              }
              else
              {
                v59 = (unsigned __int64 *)(*(_QWORD *)(a4 + 16) + v27);
                *(_DWORD *)a4 = v27 + 8;
              }
              v58 = *v59;
            }
            v117 = 0;
            LODWORD(v60) = 64;
            do
            {
              v60 = (unsigned int)(v60 - 1);
              v61 = &v116[v60];
              v116[v60] = a0123456789[v58 % 0xA + 1];
              v62 = v58;
              v58 /= 0xAu;
            }
            while ( v62 > 9 );
            v39 = 0;
            v63 = (unsigned int)(64 - v60);
            v115 = v63;
            if ( v13 != -1 && v13 > v63 )
              v39 = v13 - v63;
            if ( !v10 && v39 )
            {
              v64 = v4;
              do
              {
                v65 = v64 + 1;
                if ( v113 )
                {
                  if ( v64 < a2 )
                    *(_BYTE *)(a1 + v64) = 48;
                }
                else if ( v64 < a2 )
                {
                  *(_BYTE *)(a1 + v64) = 32;
                }
                ++v64;
              }
              while ( v65 - v4 < v39 );
              v4 += v39;
            }
            if ( v4 < a2 )
            {
              v66 = a2 - v4;
              if ( a2 - v4 > v63 )
                v66 = v63;
              qmemcpy((void *)(a1 + v4), v61, v66);
            }
            v28 = v4 + v63;
            if ( !v39 || !v10 )
              goto LABEL_259;
            for ( n = 0; n < v39; ++n )
            {
              if ( v28 + n < a2 )
                *(_BYTE *)(a1 + v28 + n) = 32;
            }
            goto LABEL_258;
        }
      }
      else if ( *v114 == 37 )
      {
        if ( v4 < a2 )
          *(_BYTE *)(a1 + v4) = 37;
        v28 = v4 + 1;
        v114 = v26 + 1;
      }
      else
      {
LABEL_173:
        v68 = *(_DWORD *)a4;
        if ( ((unsigned __int8)v24 | 0x80) == 0xBF )
        {
          if ( v68 > 0x2F )
          {
            v69 = *(unsigned int **)(a4 + 8);
            *(_QWORD *)(a4 + 8) = v69 + 2;
          }
          else
          {
            v69 = (unsigned int *)(*(_QWORD *)(a4 + 16) + v68);
            *(_DWORD *)a4 = v68 + 8;
          }
          v70 = *v69;
        }
        else
        {
          if ( v68 > 0x2F )
          {
            v71 = *(unsigned __int64 **)(a4 + 8);
            *(_QWORD *)(a4 + 8) = v71 + 1;
          }
          else
          {
            v71 = (unsigned __int64 *)(*(_QWORD *)(a4 + 16) + v68);
            *(_DWORD *)a4 = v68 + 8;
          }
          v70 = *v71;
        }
        v107 = v10;
        v72 = sub_40D550(v70, v109, *v26 == 88, (__int64)v116, &v115);
        v73 = 0;
        v74 = v72;
        if ( v13 != -1 && v13 > v115 )
          v73 = v13 - v115;
        if ( !v107 && v73 )
        {
          v75 = v4;
          do
          {
            v76 = v75 + 1;
            if ( v113 )
            {
              if ( a2 > v75 )
                *(_BYTE *)(a1 + v75) = 48;
            }
            else if ( a2 > v75 )
            {
              *(_BYTE *)(a1 + v75) = 32;
            }
            ++v75;
          }
          while ( v73 > v76 - v4 );
          v4 += v73;
        }
        v77 = v115;
        if ( v4 < a2 )
        {
          v78 = a2 - v4;
          if ( a2 - v4 > v115 )
            v78 = v115;
          qmemcpy((void *)(a1 + v4), v74, v78);
        }
        v28 = v4 + v77;
        if ( v73 && v107 )
        {
          for ( ii = 0; ii < v73; ++ii )
          {
            if ( a2 > v28 + ii )
              *(_BYTE *)(a1 + v28 + ii) = 32;
          }
LABEL_286:
          v28 += v73;
        }
LABEL_287:
        ++v114;
      }
      v4 = v28;
    }
    else
    {
      if ( v4 < a2 )
        *(_BYTE *)(a1 + v4) = v8;
      v114 = v9;
      ++v4;
    }
  }
  if ( v4 >= a2 )
    *(_BYTE *)(a1 + a2 - 1) = 0;
  else
    *(_BYTE *)(a1 + v4) = 0;
  return v4;
}
