// Function: sub_29A2B10
// Address: 0x29a2b10
//
bool __fastcall sub_29A2B10(__int64 a1)
{
  __int64 v1; // r14
  int v2; // edx
  __int64 *v3; // rax
  __int64 *v4; // r15
  __int64 v5; // r12
  __int64 v6; // rax
  __int64 v7; // r13
  _QWORD *v8; // rax
  _QWORD *v9; // rdx
  unsigned int v11; // esi
  __int64 v12; // r8
  int v13; // r11d
  __int64 *v14; // rdx
  unsigned int v15; // edi
  _QWORD *v16; // rax
  __int64 v17; // rcx
  unsigned __int64 *v18; // rdx
  int v19; // esi
  __int64 v20; // rcx
  int v21; // esi
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // rdi
  __int64 v25; // r13
  const char *v26; // rax
  __int64 v27; // rdx
  __int64 v28; // rax
  _BYTE *v29; // rax
  __int64 *v30; // rbx
  __int64 v31; // r8
  __int64 *v32; // r12
  __int64 *v33; // rbx
  __int64 v34; // r14
  const char *v35; // rax
  size_t v36; // rdx
  _WORD *v37; // rdi
  unsigned __int8 *v38; // rsi
  unsigned __int64 v39; // rax
  _BYTE *v40; // rdx
  __int64 *v41; // rbx
  unsigned __int8 *v42; // rax
  size_t v43; // rdx
  void *v44; // rdi
  __int64 v45; // rdx
  __int64 v46; // r9
  __int64 v47; // rax
  __int64 v48; // rcx
  int v49; // esi
  __int64 v50; // rdi
  __int64 v51; // r8
  int v52; // esi
  __int64 *v53; // rax
  __int64 v54; // r10
  __int64 v55; // rsi
  _QWORD *v56; // rax
  __int64 v57; // rbx
  __int64 v58; // rax
  int v59; // eax
  int v60; // eax
  unsigned __int64 v61; // rax
  int v62; // eax
  int v63; // eax
  __int64 v64; // r9
  __int64 v65; // rbx
  __int64 v66; // rdx
  const void *v67; // r13
  int v68; // ebx
  int v69; // r11d
  int v70; // r11d
  __int64 v71; // r9
  __int64 v72; // rcx
  __int64 v73; // r8
  int v74; // edi
  __int64 *v75; // rsi
  int v76; // r10d
  int v77; // r10d
  __int64 v78; // r8
  __int64 *v79; // rcx
  __int64 v80; // rbx
  int v81; // esi
  __int64 v82; // rdi
  int v83; // r8d
  __int64 v84; // [rsp+8h] [rbp-D8h]
  __int64 v85; // [rsp+10h] [rbp-D0h]
  __int64 v86; // [rsp+18h] [rbp-C8h]
  size_t v87; // [rsp+28h] [rbp-B8h]
  unsigned __int64 v88; // [rsp+30h] [rbp-B0h]
  __int64 v89; // [rsp+38h] [rbp-A8h]
  __int64 *v90; // [rsp+40h] [rbp-A0h]
  size_t v91; // [rsp+40h] [rbp-A0h]
  __int64 *v92; // [rsp+48h] [rbp-98h]
  _QWORD *v93; // [rsp+50h] [rbp-90h] BYREF
  __int64 v94; // [rsp+58h] [rbp-88h]
  _BYTE v95[16]; // [rsp+60h] [rbp-80h] BYREF
  __int64 v96[2]; // [rsp+70h] [rbp-70h] BYREF
  void *v97; // [rsp+80h] [rbp-60h]
  unsigned __int64 v98; // [rsp+88h] [rbp-58h]
  void *dest; // [rsp+90h] [rbp-50h]
  __int64 v100; // [rsp+98h] [rbp-48h]
  _QWORD **v101; // [rsp+A0h] [rbp-40h]

  v1 = a1;
  if ( *(_DWORD *)(a1 + 312) )
  {
    sub_2A3F1A0(a1 + 304);
    v65 = *(unsigned int *)(a1 + 312);
    v66 = *(unsigned int *)(a1 + 168);
    v67 = *(const void **)(a1 + 304);
    if ( v65 + v66 > (unsigned __int64)*(unsigned int *)(a1 + 172) )
    {
      sub_C8D5F0(a1 + 160, (const void *)(a1 + 176), v65 + v66, 8u, v65 + v66, v64);
      v66 = *(unsigned int *)(a1 + 168);
    }
    v3 = *(__int64 **)(a1 + 160);
    if ( 8 * v65 )
    {
      memcpy(&v3[v66], v67, 8 * v65);
      LODWORD(v66) = *(_DWORD *)(a1 + 168);
      v3 = *(__int64 **)(a1 + 160);
    }
    v68 = v66 + v65;
    *(_DWORD *)(a1 + 168) = v68;
    v2 = v68;
  }
  else
  {
    v2 = *(_DWORD *)(a1 + 168);
    v3 = *(__int64 **)(a1 + 160);
  }
  v92 = &v3[v2];
  if ( v92 != v3 )
  {
    v4 = v3;
    while ( 1 )
    {
      v5 = *v4;
      sub_AD0030(*v4);
      v6 = sub_ACADE0(*(__int64 ***)(v5 + 8));
      sub_BD84D0(v5, v6);
      v7 = *(_QWORD *)(v1 + 448);
      if ( v7 )
      {
        if ( *(_BYTE *)(v1 + 28) )
        {
          v8 = *(_QWORD **)(v1 + 8);
          v9 = &v8[*(unsigned int *)(v1 + 20)];
          if ( v8 == v9 )
            goto LABEL_17;
          while ( v5 != *v8 )
          {
            if ( v9 == ++v8 )
              goto LABEL_17;
          }
        }
        else if ( !sub_C8CA60(v1, v5) )
        {
          v7 = *(_QWORD *)(v1 + 448);
LABEL_17:
          v11 = *(_DWORD *)(v7 + 120);
          if ( v11 )
          {
            v12 = *(_QWORD *)(v7 + 104);
            v13 = 1;
            v14 = 0;
            v15 = (v11 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
            v16 = (_QWORD *)(v12 + 16LL * v15);
            v17 = *v16;
            if ( v5 == *v16 )
            {
LABEL_19:
              v18 = v16 + 1;
              v88 = v16[1];
              if ( v88 )
                goto LABEL_20;
              goto LABEL_76;
            }
            while ( v17 != -4096 )
            {
              if ( !v14 && v17 == -8192 )
                v14 = v16;
              v15 = (v11 - 1) & (v13 + v15);
              v16 = (_QWORD *)(v12 + 16LL * v15);
              v17 = *v16;
              if ( v5 == *v16 )
                goto LABEL_19;
              ++v13;
            }
            if ( !v14 )
              v14 = v16;
            v59 = *(_DWORD *)(v7 + 112);
            ++*(_QWORD *)(v7 + 96);
            v60 = v59 + 1;
            if ( 4 * v60 < 3 * v11 )
            {
              if ( v11 - *(_DWORD *)(v7 + 116) - v60 <= v11 >> 3 )
              {
                sub_D25040(v7 + 96, v11);
                v76 = *(_DWORD *)(v7 + 120);
                if ( !v76 )
                {
LABEL_119:
                  ++*(_DWORD *)(v7 + 112);
                  BUG();
                }
                v77 = v76 - 1;
                v78 = *(_QWORD *)(v7 + 104);
                v79 = 0;
                LODWORD(v80) = v77 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
                v81 = 1;
                v60 = *(_DWORD *)(v7 + 112) + 1;
                v14 = (__int64 *)(v78 + 16LL * (unsigned int)v80);
                v82 = *v14;
                if ( v5 != *v14 )
                {
                  while ( v82 != -4096 )
                  {
                    if ( v82 == -8192 && !v79 )
                      v79 = v14;
                    v80 = v77 & (unsigned int)(v80 + v81);
                    v14 = (__int64 *)(v78 + 16 * v80);
                    v82 = *v14;
                    if ( v5 == *v14 )
                      goto LABEL_73;
                    ++v81;
                  }
                  if ( v79 )
                    v14 = v79;
                }
              }
              goto LABEL_73;
            }
          }
          else
          {
            ++*(_QWORD *)(v7 + 96);
          }
          sub_D25040(v7 + 96, 2 * v11);
          v69 = *(_DWORD *)(v7 + 120);
          if ( !v69 )
            goto LABEL_119;
          v70 = v69 - 1;
          v71 = *(_QWORD *)(v7 + 104);
          LODWORD(v72) = v70 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
          v60 = *(_DWORD *)(v7 + 112) + 1;
          v14 = (__int64 *)(v71 + 16LL * (unsigned int)v72);
          v73 = *v14;
          if ( v5 != *v14 )
          {
            v74 = 1;
            v75 = 0;
            while ( v73 != -4096 )
            {
              if ( !v75 && v73 == -8192 )
                v75 = v14;
              v72 = v70 & (unsigned int)(v72 + v74);
              v14 = (__int64 *)(v71 + 16 * v72);
              v73 = *v14;
              if ( v5 == *v14 )
                goto LABEL_73;
              ++v74;
            }
            if ( v75 )
              v14 = v75;
          }
LABEL_73:
          *(_DWORD *)(v7 + 112) = v60;
          if ( *v14 != -4096 )
            --*(_DWORD *)(v7 + 116);
          *v14 = v5;
          v18 = (unsigned __int64 *)(v14 + 1);
          *v18 = 0;
LABEL_76:
          v61 = sub_D28F90((__int64 *)v7, v5, v18);
          v7 = *(_QWORD *)(v1 + 448);
          v88 = v61;
LABEL_20:
          v19 = *(_DWORD *)(v7 + 328);
          v20 = *(_QWORD *)(v7 + 312);
          if ( v19 )
          {
            v21 = v19 - 1;
            v22 = v21 & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
            v23 = (__int64 *)(v20 + 16LL * v22);
            v24 = *v23;
            if ( v88 == *v23 )
            {
LABEL_22:
              v89 = v23[1];
              goto LABEL_23;
            }
            v63 = 1;
            while ( v24 != -4096 )
            {
              v83 = v63 + 1;
              v22 = v21 & (v63 + v22);
              v23 = (__int64 *)(v20 + 16LL * v22);
              v24 = *v23;
              if ( v88 == *v23 )
                goto LABEL_22;
              v63 = v83;
            }
          }
          v89 = 0;
LABEL_23:
          v25 = *(_QWORD *)(v1 + 480);
          v26 = sub_BD5D20(v5);
          sub_BBB260(v25, v5, (__int64)v26, v27);
          v28 = *(_QWORD *)(v1 + 464);
          v95[0] = 0;
          v86 = v28;
          v94 = 0;
          v93 = v95;
          v100 = 0x100000000LL;
          v96[1] = 0;
          v97 = 0;
          v96[0] = (__int64)&unk_49DD210;
          v98 = 0;
          v101 = &v93;
          dest = 0;
          sub_CB5980((__int64)v96, 0, 0, 0);
          v29 = dest;
          if ( (unsigned __int64)dest >= v98 )
          {
            sub_CB5D20((__int64)v96, 40);
          }
          else
          {
            dest = (char *)dest + 1;
            *v29 = 40;
          }
          v30 = *(__int64 **)(v89 + 8);
          if ( v30 != &v30[*(unsigned int *)(v89 + 16)] )
          {
            v31 = *v30;
            v85 = v5;
            v32 = *(__int64 **)(v89 + 8);
            v90 = v30 + 9;
            v33 = &v30[*(unsigned int *)(v89 + 16)];
            v84 = v1;
            v34 = v31;
            while ( 1 )
            {
              v35 = sub_BD5D20(*(_QWORD *)(v34 + 8));
              v37 = dest;
              v38 = (unsigned __int8 *)v35;
              v39 = v98;
              if ( v98 - (unsigned __int64)dest < v36 )
              {
                ++v32;
                sub_CB6200((__int64)v96, v38, v36);
                v39 = v98;
                v37 = dest;
                if ( v33 == v32 )
                  goto LABEL_57;
              }
              else
              {
                if ( v36 )
                {
                  v87 = v36;
                  memcpy(dest, v38, v36);
                  v39 = v98;
                  v37 = (char *)dest + v87;
                  dest = (char *)dest + v87;
                }
                if ( v33 == ++v32 )
                {
LABEL_57:
                  v5 = v85;
                  v1 = v84;
                  if ( (unsigned __int64)v37 < v39 )
                    goto LABEL_40;
LABEL_58:
                  sub_CB5D20((__int64)v96, 41);
                  goto LABEL_41;
                }
              }
              v34 = *v32;
              if ( v39 - (unsigned __int64)v37 <= 1 )
              {
                sub_CB6200((__int64)v96, (unsigned __int8 *)", ", 2u);
              }
              else
              {
                *v37 = 8236;
                dest = (char *)dest + 2;
              }
              if ( v32 == v90 )
              {
                v40 = dest;
                v5 = v85;
                v1 = v84;
                if ( v98 - (unsigned __int64)dest <= 4 )
                {
                  v41 = (__int64 *)sub_CB6200((__int64)v96, "..., ", 5u);
                }
                else
                {
                  *(_DWORD *)dest = 741223982;
                  v41 = v96;
                  v40[4] = 32;
                  dest = (char *)dest + 5;
                }
                v42 = (unsigned __int8 *)sub_BD5D20(*(_QWORD *)(*(_QWORD *)(*(_QWORD *)(v89 + 8)
                                                                          + 8LL * *(unsigned int *)(v89 + 16)
                                                                          - 8)
                                                              + 8LL));
                v44 = (void *)v41[4];
                if ( v43 > v41[3] - (__int64)v44 )
                {
                  sub_CB6200((__int64)v41, v42, v43);
                }
                else if ( v43 )
                {
                  v91 = v43;
                  memcpy(v44, v42, v43);
                  v41[4] += v91;
                }
                break;
              }
            }
          }
          v37 = dest;
          if ( (unsigned __int64)dest >= v98 )
            goto LABEL_58;
LABEL_40:
          dest = (char *)v37 + 1;
          *(_BYTE *)v37 = 41;
LABEL_41:
          if ( dest != v97 )
            sub_CB5AE0(v96);
          v96[0] = (__int64)&unk_49DD210;
          sub_CB5840((__int64)v96);
          sub_227B210(v86, v89, (__int64)v93, v94);
          if ( v93 != (_QWORD *)v95 )
            j_j___libc_free_0((unsigned __int64)v93);
          sub_D23FF0(*(_QWORD *)(v1 + 448), v5);
          v47 = *(_QWORD *)(v1 + 448);
          v48 = *(_QWORD *)(v1 + 472);
          v49 = *(_DWORD *)(v47 + 328);
          v50 = *(_QWORD *)(v48 + 8);
          v51 = *(_QWORD *)(v47 + 312);
          if ( v49 )
          {
            v52 = v49 - 1;
            v45 = v52 & (((unsigned int)v88 >> 9) ^ ((unsigned int)v88 >> 4));
            v53 = (__int64 *)(v51 + 16 * v45);
            v54 = *v53;
            if ( v88 == *v53 )
            {
LABEL_47:
              v55 = v53[1];
              goto LABEL_48;
            }
            v62 = 1;
            while ( v54 != -4096 )
            {
              v46 = (unsigned int)(v62 + 1);
              v45 = v52 & (unsigned int)(v62 + v45);
              v53 = (__int64 *)(v51 + 16LL * (unsigned int)v45);
              v54 = *v53;
              if ( v88 == *v53 )
                goto LABEL_47;
              v62 = v46;
            }
          }
          v55 = 0;
LABEL_48:
          if ( *(_BYTE *)(v50 + 28) )
          {
            v56 = *(_QWORD **)(v50 + 8);
            v51 = *(unsigned int *)(v50 + 20);
            v45 = (__int64)&v56[v51];
            if ( v56 == (_QWORD *)v45 )
            {
LABEL_60:
              if ( (unsigned int)v51 >= *(_DWORD *)(v50 + 16) )
                goto LABEL_61;
              v51 = (unsigned int)(v51 + 1);
              *(_DWORD *)(v50 + 20) = v51;
              *(_QWORD *)v45 = v55;
              ++*(_QWORD *)v50;
              v48 = *(_QWORD *)(v1 + 472);
            }
            else
            {
              while ( v55 != *v56 )
              {
                if ( (_QWORD *)v45 == ++v56 )
                  goto LABEL_60;
              }
            }
          }
          else
          {
LABEL_61:
            sub_C8CC70(v50, v55, v45, v48, v51, v46);
            v48 = *(_QWORD *)(v1 + 472);
          }
          v57 = *(_QWORD *)(v48 + 128);
          v58 = *(unsigned int *)(v57 + 8);
          if ( v58 + 1 > (unsigned __int64)*(unsigned int *)(v57 + 12) )
          {
            sub_C8D5F0(v57, (const void *)(v57 + 16), v58 + 1, 8u, v51, v46);
            v58 = *(unsigned int *)(v57 + 8);
          }
          *(_QWORD *)(*(_QWORD *)v57 + 8 * v58) = v5;
          ++*(_DWORD *)(v57 + 8);
          goto LABEL_12;
        }
      }
      sub_B2E860((_QWORD *)v5);
LABEL_12:
      if ( v92 == ++v4 )
      {
        v2 = *(_DWORD *)(v1 + 168);
        break;
      }
    }
  }
  *(_DWORD *)(v1 + 312) = 0;
  *(_DWORD *)(v1 + 168) = 0;
  return v2 != 0;
}
