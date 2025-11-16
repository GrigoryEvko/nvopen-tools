// Function: sub_2BA36A0
// Address: 0x2ba36a0
//
void __fastcall sub_2BA36A0(__int64 a1, __int64 a2, char a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rdi
  unsigned int v8; // eax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rax
  _DWORD *v12; // r15
  __int64 v13; // r14
  char *v14; // rdi
  __int64 v15; // rbx
  __int64 v16; // rdx
  int v17; // eax
  __int64 v18; // rsi
  int v19; // edi
  unsigned int v20; // ecx
  __int64 *v21; // rax
  __int64 v22; // r8
  __int64 v23; // rax
  __int64 v24; // r12
  __int64 v25; // rax
  unsigned __int64 v26; // rdx
  _QWORD *v27; // r12
  char v28; // al
  __int64 v29; // rax
  __int64 v30; // rdx
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // rcx
  __int64 v34; // rdx
  __int64 v35; // rsi
  __int64 v36; // rdx
  __int64 v37; // rbx
  char v38; // al
  unsigned int v39; // r14d
  unsigned int v40; // r12d
  __int64 v41; // r13
  char v42; // bl
  unsigned __int64 v43; // r8
  __int64 v44; // rax
  __int64 v45; // rax
  __int64 v46; // rdx
  __int64 v47; // rsi
  __int64 v48; // rax
  int v49; // ecx
  int v50; // edx
  int v51; // eax
  __int64 v52; // rbx
  __int64 v53; // rbx
  __int64 v54; // rdx
  __int64 v55; // rsi
  int v56; // eax
  __int64 v57; // rcx
  int v58; // esi
  unsigned int v59; // edx
  __int64 *v60; // rax
  __int64 v61; // rdi
  __int64 v62; // r13
  __int64 v63; // rdx
  __int64 v64; // r8
  __int64 v65; // r13
  __int64 v66; // rax
  unsigned __int64 v67; // rdx
  __int64 v68; // rbx
  __int64 v69; // rbx
  __int64 v70; // rax
  __int64 v71; // rdx
  __int64 v72; // rdx
  int v73; // eax
  __int64 v74; // rcx
  int v75; // esi
  unsigned int v76; // edx
  __int64 *v77; // rax
  __int64 v78; // rdi
  __int64 v79; // r12
  __int64 v80; // rdx
  __int64 v81; // r8
  __int64 v82; // r12
  __int64 v83; // rax
  unsigned __int64 v84; // rdx
  int v85; // eax
  int v86; // r8d
  int v87; // esi
  __int64 v88; // rdi
  int v89; // esi
  unsigned int v90; // ecx
  __int64 *v91; // rdx
  __int64 v92; // r8
  __int64 v93; // rbx
  __int64 v94; // rdx
  __int64 v95; // r8
  __int64 v96; // rbx
  __int64 v97; // rax
  unsigned __int64 v98; // rdx
  int v99; // eax
  int v100; // r8d
  int v101; // edx
  __int64 v102; // [rsp+0h] [rbp-110h]
  __int64 v104; // [rsp+20h] [rbp-F0h]
  __int64 v106; // [rsp+38h] [rbp-D8h]
  __int64 v107; // [rsp+48h] [rbp-C8h] BYREF
  __m128i v108; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v109; // [rsp+60h] [rbp-B0h]
  __int64 v110; // [rsp+68h] [rbp-A8h]
  __int64 v111; // [rsp+70h] [rbp-A0h]
  __int64 v112; // [rsp+78h] [rbp-98h]
  _QWORD *v113; // [rsp+80h] [rbp-90h] BYREF
  __int64 v114; // [rsp+88h] [rbp-88h]
  _QWORD v115[16]; // [rsp+90h] [rbp-80h] BYREF

  v7 = v115;
  v115[0] = a2;
  v113 = v115;
  v114 = 0xA00000001LL;
  v8 = 1;
  do
  {
    v9 = v8--;
    v10 = v7[v9 - 1];
    LODWORD(v114) = v8;
    v107 = v10;
    if ( !v10 )
    {
      if ( a3 )
        goto LABEL_155;
      continue;
    }
    v11 = a1;
    v12 = (_DWORD *)v10;
    v13 = v11;
    do
    {
LABEL_5:
      if ( v12[36] != -1 )
        goto LABEL_4;
      *((_QWORD *)v12 + 18) = 0;
      v14 = *(char **)v12;
      v15 = *(_QWORD *)(*(_QWORD *)v12 + 16LL);
      if ( !v15 )
        goto LABEL_21;
      do
      {
        while ( 1 )
        {
          v16 = *(_QWORD *)(v15 + 24);
          if ( *(_QWORD *)v13 == *(_QWORD *)(v16 + 40) )
          {
            v17 = *(_DWORD *)(v13 + 104);
            v18 = *(_QWORD *)(v13 + 88);
            if ( v17 )
            {
              v19 = v17 - 1;
              v20 = (v17 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
              v21 = (__int64 *)(v18 + 16LL * v20);
              v22 = *v21;
              if ( v16 != *v21 )
              {
                v51 = 1;
                while ( v22 != -4096 )
                {
                  a6 = (unsigned int)(v51 + 1);
                  v20 = v19 & (v51 + v20);
                  v21 = (__int64 *)(v18 + 16LL * v20);
                  v22 = *v21;
                  if ( v16 == *v21 )
                    goto LABEL_12;
                  v51 = a6;
                }
                goto LABEL_8;
              }
LABEL_12:
              v23 = v21[1];
              if ( v23 && *(_DWORD *)(v23 + 136) == *(_DWORD *)(v13 + 204) )
              {
                ++v12[36];
                v24 = *(_QWORD *)(v23 + 16);
                if ( !*(_BYTE *)(v24 + 152) )
                  ++v12[37];
                if ( *(_DWORD *)(v24 + 144) == -1 )
                  break;
              }
            }
          }
LABEL_8:
          v15 = *(_QWORD *)(v15 + 8);
          if ( !v15 )
            goto LABEL_20;
        }
        v25 = (unsigned int)v114;
        v26 = (unsigned int)v114 + 1LL;
        if ( v26 > HIDWORD(v114) )
        {
          sub_C8D5F0((__int64)&v113, v115, v26, 8u, v22, a6);
          v25 = (unsigned int)v114;
        }
        v113[v25] = v24;
        LODWORD(v114) = v114 + 1;
        v15 = *(_QWORD *)(v15 + 8);
      }
      while ( v15 );
LABEL_20:
      v14 = *(char **)v12;
LABEL_21:
      if ( !(unsigned __int8)sub_98CD80(v14) )
      {
        v52 = *(_QWORD *)(*(_QWORD *)v12 + 32LL);
        if ( v52 == *(_QWORD *)(*(_QWORD *)v12 + 40LL) + 48LL || !v52 )
          v53 = 0;
        else
          v53 = v52 - 24;
        if ( *(_QWORD *)(v13 + 168) != v53 )
        {
          while ( 1 )
          {
            v55 = *(_QWORD *)(*(_QWORD *)v13 + 56LL);
            if ( v55 )
              LODWORD(v55) = v55 - 24;
            if ( !sub_991A70((unsigned __int8 *)v53, v55, *(_QWORD *)(a4 + 3328), 0, 0, 1u, 0) )
              break;
LABEL_92:
            v54 = *(_QWORD *)(v53 + 32);
            if ( v54 == *(_QWORD *)(v53 + 40) + 48LL || (v53 = v54 - 24, !v54) )
              v53 = 0;
            if ( *(_QWORD *)(v13 + 168) == v53 )
              goto LABEL_22;
          }
          if ( *(_QWORD *)v13 != *(_QWORD *)(v53 + 40) )
            goto LABEL_187;
          v56 = *(_DWORD *)(v13 + 104);
          v57 = *(_QWORD *)(v13 + 88);
          if ( !v56 )
            goto LABEL_187;
          v58 = v56 - 1;
          v59 = (v56 - 1) & (((unsigned int)v53 >> 9) ^ ((unsigned int)v53 >> 4));
          v60 = (__int64 *)(v57 + 16LL * v59);
          v61 = *v60;
          if ( v53 != *v60 )
          {
            v85 = 1;
            while ( v61 != -4096 )
            {
              v86 = v85 + 1;
              v59 = v58 & (v85 + v59);
              v60 = (__int64 *)(v57 + 16LL * v59);
              v61 = *v60;
              if ( v53 == *v60 )
                goto LABEL_101;
              v85 = v86;
            }
LABEL_187:
            BUG();
          }
LABEL_101:
          v62 = v60[1];
          if ( !v62 || *(_DWORD *)(v62 + 136) != *(_DWORD *)(v13 + 204) )
            goto LABEL_187;
          v63 = *(unsigned int *)(v62 + 96);
          v64 = v63 + 1;
          if ( v63 + 1 > (unsigned __int64)*(unsigned int *)(v62 + 100) )
          {
            sub_C8D5F0(v62 + 88, (const void *)(v62 + 104), v63 + 1, 8u, v64, a6);
            v63 = *(unsigned int *)(v62 + 96);
          }
          *(_QWORD *)(*(_QWORD *)(v62 + 88) + 8 * v63) = v12;
          ++*(_DWORD *)(v62 + 96);
          ++v12[36];
          v65 = *(_QWORD *)(v62 + 16);
          if ( *(_BYTE *)(v65 + 152) )
          {
            if ( *(_DWORD *)(v65 + 144) == -1 )
              goto LABEL_107;
          }
          else
          {
            ++v12[37];
            if ( *(_DWORD *)(v65 + 144) == -1 )
            {
LABEL_107:
              v66 = (unsigned int)v114;
              v67 = (unsigned int)v114 + 1LL;
              if ( v67 > HIDWORD(v114) )
              {
                sub_C8D5F0((__int64)&v113, v115, v67, 8u, v64, a6);
                v66 = (unsigned int)v114;
              }
              v113[v66] = v65;
              LODWORD(v114) = v114 + 1;
            }
          }
          if ( !(unsigned __int8)sub_98CD80((char *)v53) )
            goto LABEL_22;
          goto LABEL_92;
        }
      }
LABEL_22:
      if ( !*(_BYTE *)(v13 + 192) )
        goto LABEL_42;
      v27 = *(_QWORD **)v12;
      v28 = **(_BYTE **)v12;
      if ( v28 == 85 )
      {
        v29 = *(v27 - 4);
        if ( !v29
          || *(_BYTE *)v29
          || ((v30 = *(_QWORD *)(v29 + 24), v30 != v27[10]) || *(_DWORD *)(v29 + 36) != 343)
          && (v30 != v27[10] || *(_DWORD *)(v29 + 36) != 342)
          || ((v68 = v27[4], v68 == v27[5] + 48LL) || !v68 ? (v69 = 0) : (v69 = v68 - 24), v69 == *(_QWORD *)(v13 + 168)) )
        {
          if ( (unsigned __int8)sub_B46420(*(_QWORD *)v12) )
            goto LABEL_29;
LABEL_77:
          if ( (unsigned __int8)sub_B46490((__int64)v27) )
            goto LABEL_29;
          goto LABEL_42;
        }
        do
        {
          if ( *(_BYTE *)v69 == 85 )
          {
            v70 = *(_QWORD *)(v69 - 32);
            if ( v70 && !*(_BYTE *)v70 )
            {
              v71 = *(_QWORD *)(v70 + 24);
              if ( v71 == *(_QWORD *)(v69 + 80) && *(_DWORD *)(v70 + 36) == 343 )
                break;
              if ( v71 == *(_QWORD *)(v69 + 80) && *(_DWORD *)(v70 + 36) == 342 )
                break;
            }
          }
          else if ( *(_BYTE *)v69 == 60 )
          {
            if ( *(_QWORD *)(v69 + 40) != *(_QWORD *)v13 )
              goto LABEL_186;
            v73 = *(_DWORD *)(v13 + 104);
            v74 = *(_QWORD *)(v13 + 88);
            if ( !v73 )
              goto LABEL_186;
            v75 = v73 - 1;
            v76 = (v73 - 1) & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
            v77 = (__int64 *)(v74 + 16LL * v76);
            v78 = *v77;
            if ( *v77 != v69 )
            {
              v99 = 1;
              while ( v78 != -4096 )
              {
                v100 = v99 + 1;
                v76 = v75 & (v99 + v76);
                v77 = (__int64 *)(v74 + 16LL * v76);
                v78 = *v77;
                if ( *v77 == v69 )
                  goto LABEL_134;
                v99 = v100;
              }
LABEL_186:
              BUG();
            }
LABEL_134:
            v79 = v77[1];
            if ( !v79 || *(_DWORD *)(v79 + 136) != *(_DWORD *)(v13 + 204) )
              goto LABEL_186;
            v80 = *(unsigned int *)(v79 + 96);
            v81 = v80 + 1;
            if ( v80 + 1 > (unsigned __int64)*(unsigned int *)(v79 + 100) )
            {
              sub_C8D5F0(v79 + 88, (const void *)(v79 + 104), v80 + 1, 8u, v81, a6);
              v80 = *(unsigned int *)(v79 + 96);
            }
            *(_QWORD *)(*(_QWORD *)(v79 + 88) + 8 * v80) = v12;
            ++*(_DWORD *)(v79 + 96);
            ++v12[36];
            v82 = *(_QWORD *)(v79 + 16);
            if ( !*(_BYTE *)(v82 + 152) )
              ++v12[37];
            if ( *(_DWORD *)(v82 + 144) == -1 )
            {
              v83 = (unsigned int)v114;
              v84 = (unsigned int)v114 + 1LL;
              if ( v84 > HIDWORD(v114) )
              {
                sub_C8D5F0((__int64)&v113, v115, v84, 8u, v81, a6);
                v83 = (unsigned int)v114;
              }
              v113[v83] = v82;
              LODWORD(v114) = v114 + 1;
            }
          }
          v72 = *(_QWORD *)(v69 + 32);
          if ( v72 == *(_QWORD *)(v69 + 40) + 48LL || !v72 )
            v69 = 0;
          else
            v69 = v72 - 24;
        }
        while ( *(_QWORD *)(v13 + 168) != v69 );
        v27 = *(_QWORD **)v12;
        v28 = **(_BYTE **)v12;
      }
      if ( v28 != 60 && !(unsigned __int8)sub_B46420((__int64)v27) )
        goto LABEL_77;
LABEL_29:
      v31 = v27[4];
      if ( v31 == v27[5] + 48LL || !v31 )
        v32 = 0;
      else
        v32 = v31 - 24;
      v33 = *(_QWORD *)(v13 + 168);
      if ( v33 != v32 )
      {
        while ( 1 )
        {
          if ( *(_BYTE *)v32 == 85 )
          {
            v34 = *(_QWORD *)(v32 - 32);
            if ( v34 )
            {
              if ( !*(_BYTE *)v34 )
              {
                v35 = *(_QWORD *)(v34 + 24);
                if ( v35 == *(_QWORD *)(v32 + 80) && *(_DWORD *)(v34 + 36) == 343 )
                  break;
                if ( v35 == *(_QWORD *)(v32 + 80) && *(_DWORD *)(v34 + 36) == 342 )
                  break;
              }
            }
          }
          v36 = *(_QWORD *)(v32 + 32);
          if ( v36 == *(_QWORD *)(v32 + 40) + 48LL || !v36 )
            v32 = 0;
          else
            v32 = v36 - 24;
          if ( v33 == v32 )
            goto LABEL_42;
        }
        if ( *(_QWORD *)v13 == *(_QWORD *)(v32 + 40) )
        {
          v87 = *(_DWORD *)(v13 + 104);
          v88 = *(_QWORD *)(v13 + 88);
          if ( v87 )
          {
            v89 = v87 - 1;
            v90 = v89 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
            v91 = (__int64 *)(v88 + 16LL * v90);
            v92 = *v91;
            if ( *v91 == v32 )
            {
LABEL_160:
              v93 = v91[1];
              if ( v93 && *(_DWORD *)(v93 + 136) == *(_DWORD *)(v13 + 204) )
              {
                v94 = *(unsigned int *)(v93 + 96);
                v95 = v94 + 1;
                if ( v94 + 1 > (unsigned __int64)*(unsigned int *)(v93 + 100) )
                {
                  sub_C8D5F0(v93 + 88, (const void *)(v93 + 104), v94 + 1, 8u, v95, a6);
                  v94 = *(unsigned int *)(v93 + 96);
                }
                *(_QWORD *)(*(_QWORD *)(v93 + 88) + 8 * v94) = v12;
                ++*(_DWORD *)(v93 + 96);
                ++v12[36];
                v96 = *(_QWORD *)(v93 + 16);
                if ( !*(_BYTE *)(v96 + 152) )
                  ++v12[37];
                if ( *(_DWORD *)(v96 + 144) == -1 )
                {
                  v97 = (unsigned int)v114;
                  v98 = (unsigned int)v114 + 1LL;
                  if ( v98 > HIDWORD(v114) )
                  {
                    sub_C8D5F0((__int64)&v113, v115, v98, 8u, v95, a6);
                    v97 = (unsigned int)v114;
                  }
                  v113[v97] = v96;
                  LODWORD(v114) = v114 + 1;
                }
                goto LABEL_42;
              }
            }
            else
            {
              v101 = 1;
              while ( v92 != -4096 )
              {
                a6 = (unsigned int)(v101 + 1);
                v90 = v89 & (v101 + v90);
                v91 = (__int64 *)(v88 + 16LL * v90);
                v92 = *v91;
                if ( v32 == *v91 )
                  goto LABEL_160;
                v101 = a6;
              }
            }
          }
        }
        BUG();
      }
LABEL_42:
      v37 = *((_QWORD *)v12 + 4);
      if ( !v37 )
      {
LABEL_4:
        v12 = (_DWORD *)*((_QWORD *)v12 + 3);
        if ( !v12 )
          break;
        goto LABEL_5;
      }
      v38 = **(_BYTE **)v12;
      v106 = *(_QWORD *)v12;
      if ( v38 == 62 )
      {
        sub_D66630(&v108, *(_QWORD *)v12);
      }
      else if ( v38 == 61 )
      {
        sub_D665A0(&v108, v106);
      }
      else
      {
        v108.m128i_i64[0] = 0;
        v108.m128i_i64[1] = -1;
        v109 = 0;
        v110 = 0;
        v111 = 0;
        v112 = 0;
      }
      v104 = v13;
      v39 = 1;
      v40 = 0;
      v41 = v37;
      v42 = sub_B46490(*(_QWORD *)v12);
      do
      {
        if ( v39 > 0x9F
          || (v42 || (unsigned __int8)sub_B46490(*(_QWORD *)v41))
          && (v40 > 9 || (unsigned __int8)sub_2B4C6A0(a4, &v108, v106, *(unsigned __int8 **)v41)) )
        {
          v44 = *(unsigned int *)(v41 + 48);
          ++v40;
          if ( v44 + 1 > (unsigned __int64)*(unsigned int *)(v41 + 52) )
          {
            sub_C8D5F0(v41 + 40, (const void *)(v41 + 56), v44 + 1, 8u, v43, a6);
            v44 = *(unsigned int *)(v41 + 48);
          }
          *(_QWORD *)(*(_QWORD *)(v41 + 40) + 8 * v44) = v12;
          ++*(_DWORD *)(v41 + 48);
          ++v12[36];
          v45 = *(_QWORD *)(v41 + 16);
          if ( *(_BYTE *)(v45 + 152) )
          {
            if ( *(_DWORD *)(v45 + 144) != -1 )
              goto LABEL_47;
          }
          else
          {
            ++v12[37];
            if ( *(_DWORD *)(v45 + 144) != -1 )
            {
LABEL_47:
              if ( v39 == 320 )
                break;
              goto LABEL_48;
            }
          }
          v46 = (unsigned int)v114;
          v43 = (unsigned int)v114 + 1LL;
          if ( v43 > HIDWORD(v114) )
          {
            v102 = v45;
            sub_C8D5F0((__int64)&v113, v115, (unsigned int)v114 + 1LL, 8u, v43, a6);
            v46 = (unsigned int)v114;
            v45 = v102;
          }
          v113[v46] = v45;
          LODWORD(v114) = v114 + 1;
          if ( v39 == 320 )
            break;
        }
LABEL_48:
        v41 = *(_QWORD *)(v41 + 32);
        ++v39;
      }
      while ( v41 );
      v12 = (_DWORD *)*((_QWORD *)v12 + 3);
      v13 = v104;
    }
    while ( v12 );
    a1 = v13;
    if ( !a3 )
      goto LABEL_66;
    v47 = v107;
    if ( v107 )
    {
      v48 = v107;
      v49 = 0;
      while ( 1 )
      {
        v50 = *(_DWORD *)(v48 + 148);
        if ( v50 == -1 )
          goto LABEL_66;
        v48 = *(_QWORD *)(v48 + 24);
        v49 += v50;
        if ( !v48 )
        {
          if ( v49 )
            goto LABEL_66;
          goto LABEL_112;
        }
      }
    }
LABEL_155:
    v47 = 0;
LABEL_112:
    if ( !*(_BYTE *)(v47 + 152) )
      sub_2BA3420(a1 + 112, &v107);
LABEL_66:
    v8 = v114;
    v7 = v113;
  }
  while ( v8 );
  if ( v7 != v115 )
    _libc_free((unsigned __int64)v7);
}
