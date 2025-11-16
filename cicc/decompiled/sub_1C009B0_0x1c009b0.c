// Function: sub_1C009B0
// Address: 0x1c009b0
//
__int64 __fastcall sub_1C009B0(__int64 a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  int v5; // edx
  int v6; // edx
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rsi
  __int64 v11; // r10
  __int64 *v12; // rbx
  __int64 v13; // rax
  __int64 v14; // r13
  int v15; // ecx
  __int64 *v16; // rdx
  __int64 v17; // rax
  __int64 *v18; // r12
  unsigned int v19; // esi
  __int64 v20; // rcx
  __int64 v21; // r8
  unsigned int v22; // edx
  __int64 *v23; // rax
  __int64 v24; // rdi
  __int64 v26; // rsi
  __int64 v27; // r13
  int v28; // r8d
  __int64 v29; // rdx
  unsigned int v30; // r14d
  unsigned int v31; // edi
  __int64 *v32; // rax
  __int64 v33; // rcx
  _BYTE *v34; // rsi
  __int64 v35; // rax
  __int64 v36; // rsi
  int v37; // edi
  __int64 v38; // rdx
  int v39; // r8d
  unsigned int v40; // r14d
  unsigned int v41; // r10d
  __int64 *v42; // rax
  __int64 v43; // rcx
  __int64 v44; // r9
  int v45; // ecx
  _DWORD *v46; // rdi
  int v47; // r8d
  int v48; // r11d
  __int64 v49; // r10
  unsigned int v50; // ebx
  int v51; // r9d
  unsigned int j; // edx
  unsigned int v53; // eax
  unsigned int *v54; // rsi
  unsigned int v55; // ecx
  unsigned int v56; // edx
  unsigned int v57; // edx
  int v58; // edx
  __int64 v59; // r9
  int i; // r10d
  _BYTE *v61; // rsi
  int v62; // r9d
  __int64 *v63; // r8
  int v64; // eax
  int v65; // r9d
  __int64 v66; // r10
  __int64 *v67; // r10
  __int64 *v68; // r10
  int v69; // r9d
  int v70; // edi
  int v71; // edx
  __int64 v72; // rdx
  __int64 v73; // rax
  int v74; // edx
  __int64 v75; // r14
  int v76; // esi
  __int64 *v77; // rcx
  __int64 *v78; // rsi
  int v79; // r14d
  int v80; // edx
  __int64 v81; // rcx
  __int64 *v82; // r8
  int v83; // r9d
  __int64 *v84; // r8
  int v85; // edi
  __int64 v86; // r14
  int v87; // ecx
  int v88; // r11d
  __int64 *v89; // r10
  int v90; // edi
  int v91; // r10d
  int v92; // r10d
  __int64 v93; // r8
  unsigned int v94; // ecx
  __int64 v95; // rsi
  int v96; // r9d
  __int64 *v97; // rdi
  int v98; // eax
  int v99; // r9d
  __int64 v100; // r10
  unsigned int v101; // r8d
  __int64 *v102; // rcx
  int v103; // esi
  __int64 v104; // rdi
  __int64 *v105; // rdx
  int v106; // ebx
  __int64 v107; // rdx
  int v108; // [rsp+4h] [rbp-ACh]
  __int64 v109; // [rsp+8h] [rbp-A8h]
  __int64 v110; // [rsp+8h] [rbp-A8h]
  int v111; // [rsp+10h] [rbp-A0h]
  __int64 v112; // [rsp+10h] [rbp-A0h]
  __int64 v113; // [rsp+10h] [rbp-A0h]
  __int64 v114; // [rsp+10h] [rbp-A0h]
  __int64 v115; // [rsp+10h] [rbp-A0h]
  __int64 v116; // [rsp+10h] [rbp-A0h]
  int v117; // [rsp+10h] [rbp-A0h]
  __int64 v118; // [rsp+10h] [rbp-A0h]
  __int64 v119; // [rsp+10h] [rbp-A0h]
  __int64 v120; // [rsp+10h] [rbp-A0h]
  unsigned int v122; // [rsp+20h] [rbp-90h]
  __int64 v123; // [rsp+28h] [rbp-88h] BYREF
  __int64 v124; // [rsp+30h] [rbp-80h] BYREF
  __int64 *v125; // [rsp+38h] [rbp-78h] BYREF
  _DWORD *v126; // [rsp+40h] [rbp-70h] BYREF
  _BYTE *v127; // [rsp+48h] [rbp-68h]
  _BYTE *v128; // [rsp+50h] [rbp-60h]
  __int64 v129; // [rsp+60h] [rbp-50h] BYREF
  __int64 *v130; // [rsp+68h] [rbp-48h]
  __int64 v131; // [rsp+70h] [rbp-40h]
  __int64 v132; // [rsp+78h] [rbp-38h]

  v131 = 0;
  v132 = 0;
  v5 = *(_DWORD *)(a2 + 20);
  v123 = a2;
  v129 = 0;
  v130 = 0;
  v6 = v5 & 0xFFFFFFF;
  if ( !v6 )
  {
    v126 = 0;
    v127 = 0;
    v128 = 0;
LABEL_12:
    v122 = 0;
    goto LABEL_13;
  }
  v8 = a2;
  v9 = 0;
  v10 = 0;
  v11 = 24LL * (unsigned int)(v6 - 1);
  v12 = 0;
  while ( 1 )
  {
    if ( (*(_BYTE *)(v8 + 23) & 0x40) != 0 )
      v13 = *(_QWORD *)(v8 - 8);
    else
      v13 = v8 - 24LL * (*(_DWORD *)(v8 + 20) & 0xFFFFFFF);
    v14 = *(_QWORD *)(v13 + v9);
    if ( *(_BYTE *)(v14 + 16) <= 0x17u )
      goto LABEL_7;
    if ( (_DWORD)v10 )
    {
      v15 = (v10 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
      v16 = &v12[v15];
      v17 = *v16;
      if ( *v16 == v14 )
        goto LABEL_7;
      v62 = 1;
      v63 = 0;
      while ( v17 != -8 )
      {
        if ( v63 || v17 != -16 )
          v16 = v63;
        v15 = (v10 - 1) & (v62 + v15);
        v17 = v12[v15];
        if ( v17 == v14 )
          goto LABEL_7;
        ++v62;
        v63 = v16;
        v16 = &v12[v15];
      }
      if ( !v63 )
        v63 = v16;
      ++v129;
      v64 = v131 + 1;
      if ( 4 * ((int)v131 + 1) < (unsigned int)(3 * v10) )
      {
        if ( (int)v10 - (v64 + HIDWORD(v131)) <= (unsigned int)v10 >> 3 )
        {
          v110 = v11;
          v116 = a1;
          sub_1467110((__int64)&v129, v10);
          if ( !(_DWORD)v132 )
          {
LABEL_185:
            LODWORD(v131) = v131 + 1;
            BUG();
          }
          v78 = 0;
          a1 = v116;
          v79 = (v132 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
          v11 = v110;
          v80 = 1;
          v63 = &v130[v79];
          v81 = *v63;
          v64 = v131 + 1;
          if ( *v63 != v14 )
          {
            while ( v81 != -8 )
            {
              if ( v81 == -16 && !v78 )
                v78 = v63;
              v106 = v80 + 1;
              v107 = ((_DWORD)v132 - 1) & (unsigned int)(v79 + v80);
              v63 = &v130[v107];
              v79 = v107;
              v81 = *v63;
              if ( *v63 == v14 )
                goto LABEL_73;
              v80 = v106;
            }
            if ( v78 )
              v63 = v78;
          }
        }
        goto LABEL_73;
      }
    }
    else
    {
      ++v129;
    }
    v109 = v11;
    v115 = a1;
    sub_1467110((__int64)&v129, 2 * v10);
    if ( !(_DWORD)v132 )
      goto LABEL_185;
    a1 = v115;
    v11 = v109;
    v74 = (v132 - 1) & (((unsigned int)v14 >> 9) ^ ((unsigned int)v14 >> 4));
    v63 = &v130[v74];
    v75 = *v63;
    v64 = v131 + 1;
    if ( *v63 != v14 )
    {
      v76 = 1;
      v77 = 0;
      while ( v75 != -8 )
      {
        if ( v75 == -16 && !v77 )
          v77 = v63;
        v74 = (v132 - 1) & (v74 + v76);
        v63 = &v130[v74];
        v75 = *v63;
        if ( *v63 == v14 )
          goto LABEL_73;
        ++v76;
      }
      if ( v77 )
        v63 = v77;
    }
LABEL_73:
    LODWORD(v131) = v64;
    if ( *v63 != -8 )
      --HIDWORD(v131);
    *v63 = v14;
    v12 = v130;
    v10 = (unsigned int)v132;
LABEL_7:
    if ( v9 == v11 )
      break;
    v8 = v123;
    v9 += 24;
  }
  v126 = 0;
  v18 = &v12[v10];
  v127 = 0;
  v128 = 0;
  if ( !(_DWORD)v131 || v12 == v18 )
    goto LABEL_12;
  while ( *v12 == -8 || *v12 == -16 )
  {
    if ( ++v12 == v18 )
      goto LABEL_12;
  }
  if ( v12 == v18 )
    goto LABEL_12;
  v122 = 0;
  do
  {
    v26 = *(unsigned int *)(a3 + 24);
    if ( (_DWORD)v26 )
    {
      v27 = *v12;
      v28 = v26 - 1;
      v29 = *(_QWORD *)(a3 + 8);
      v30 = ((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4);
      v31 = (v26 - 1) & v30;
      v32 = (__int64 *)(v29 + 16LL * v31);
      v33 = *v32;
      if ( *v12 == *v32 )
      {
        if ( v32 != (__int64 *)(v29 + 16 * v26) )
          goto LABEL_28;
      }
      else
      {
        v111 = (v26 - 1) & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4));
        v59 = *v32;
        for ( i = 1; ; i = v108 )
        {
          if ( v59 == -8 )
            goto LABEL_61;
          v65 = i + 1;
          v66 = v28 & (unsigned int)(v111 + i);
          v108 = v65;
          v111 = v66;
          v67 = (__int64 *)(v29 + 16 * v66);
          v59 = *v67;
          if ( v27 == *v67 )
            break;
        }
        if ( v67 != (__int64 *)(v29 + 16LL * (unsigned int)v26) )
        {
          v114 = *(_QWORD *)(a3 + 8);
          v68 = (__int64 *)(v29 + 16LL * (((_DWORD)v26 - 1) & (((unsigned int)*v12 >> 9) ^ ((unsigned int)*v12 >> 4))));
          v69 = 1;
          v32 = 0;
          while ( v33 != -8 )
          {
            if ( v32 || v33 != -16 )
              v68 = v32;
            v31 = v28 & (v69 + v31);
            v32 = (__int64 *)(v114 + 16LL * v31);
            v33 = *v32;
            if ( v27 == *v32 )
              goto LABEL_28;
            ++v69;
            v105 = v68;
            v68 = (__int64 *)(v114 + 16LL * v31);
            v32 = v105;
          }
          v70 = *(_DWORD *)(a3 + 16);
          if ( !v32 )
            v32 = v68;
          ++*(_QWORD *)a3;
          v71 = v70 + 1;
          if ( 4 * (v70 + 1) >= (unsigned int)(3 * v26) )
          {
            v118 = a1;
            sub_14672C0(a3, 2 * v26);
            v91 = *(_DWORD *)(a3 + 24);
            if ( v91 )
            {
              v92 = v91 - 1;
              v93 = *(_QWORD *)(a3 + 8);
              v94 = v92 & v30;
              a1 = v118;
              v71 = *(_DWORD *)(a3 + 16) + 1;
              v32 = (__int64 *)(v93 + 16LL * (v92 & v30));
              v95 = *v32;
              if ( v27 != *v32 )
              {
                v96 = 1;
                v97 = 0;
                while ( v95 != -8 )
                {
                  if ( v95 == -16 && !v97 )
                    v97 = v32;
                  v94 = v92 & (v96 + v94);
                  v32 = (__int64 *)(v93 + 16LL * v94);
                  v95 = *v32;
                  if ( v27 == *v32 )
                    goto LABEL_84;
                  ++v96;
                }
                if ( v97 )
                  v32 = v97;
              }
              goto LABEL_84;
            }
          }
          else
          {
            if ( (int)v26 - *(_DWORD *)(a3 + 20) - v71 > (unsigned int)v26 >> 3 )
            {
LABEL_84:
              *(_DWORD *)(a3 + 16) = v71;
              if ( *v32 != -8 )
                --*(_DWORD *)(a3 + 20);
              *v32 = v27;
              *((_DWORD *)v32 + 2) = 0;
LABEL_28:
              v34 = v127;
              if ( v127 == v128 )
              {
                v112 = a1;
                sub_B8BBF0((__int64)&v126, v127, (_DWORD *)v32 + 2);
                a1 = v112;
              }
              else
              {
                if ( v127 )
                {
                  *(_DWORD *)v127 = *((_DWORD *)v32 + 2);
                  v34 = v127;
                }
                v127 = v34 + 4;
              }
              v35 = *(_QWORD *)(v27 + 8);
              if ( v35 && !*(_QWORD *)(v35 + 8) )
                goto LABEL_40;
              if ( !a4 )
                goto LABEL_39;
              v36 = *(unsigned int *)(a1 + 32);
              v124 = v27;
              if ( !(_DWORD)v36 )
                goto LABEL_39;
              v37 = v36 - 1;
              v38 = *(_QWORD *)(a1 + 16);
              v39 = 1;
              v40 = (v36 - 1) & v30;
              v41 = v40;
              v42 = (__int64 *)(v38 + 16LL * v40);
              v43 = *v42;
              v44 = *v42;
              if ( v27 == *v42 )
              {
                if ( v42 != (__int64 *)(16 * v36 + v38) )
                {
LABEL_38:
                  v45 = *((_DWORD *)v42 + 2);
                  if ( v45 < 0 )
                    goto LABEL_39;
                  v72 = 1LL << v45;
                  v73 = 8LL * ((unsigned int)v45 >> 6);
LABEL_88:
                  if ( (*(_QWORD *)(*a4 + v73) & v72) != 0 )
                    goto LABEL_40;
                }
LABEL_39:
                ++v122;
                goto LABEL_40;
              }
              while ( 1 )
              {
                if ( v44 == -8 )
                  goto LABEL_39;
                v41 = v37 & (v39 + v41);
                v117 = v39 + 1;
                v82 = (__int64 *)(v38 + 16LL * v41);
                v44 = *v82;
                if ( v27 == *v82 )
                  break;
                v39 = v117;
              }
              if ( v82 == (__int64 *)(v38 + 16LL * (unsigned int)v36) )
                goto LABEL_39;
              v83 = 1;
              v84 = 0;
              while ( v43 != -8 )
              {
                if ( v43 == -16 && !v84 )
                  v84 = v42;
                v40 = v37 & (v83 + v40);
                v42 = (__int64 *)(v38 + 16LL * v40);
                v43 = *v42;
                if ( v27 == *v42 )
                  goto LABEL_38;
                ++v83;
              }
              v85 = *(_DWORD *)(a1 + 24);
              v86 = a1 + 8;
              if ( v84 )
                v42 = v84;
              ++*(_QWORD *)(a1 + 8);
              v87 = v85 + 1;
              if ( 4 * (v85 + 1) >= (unsigned int)(3 * v36) )
              {
                v120 = a1;
                LODWORD(v36) = 2 * v36;
              }
              else
              {
                if ( (int)v36 - *(_DWORD *)(a1 + 28) - v87 > (unsigned int)v36 >> 3 )
                {
LABEL_114:
                  *(_DWORD *)(a1 + 24) = v87;
                  if ( *v42 != -8 )
                    --*(_DWORD *)(a1 + 28);
                  *v42 = v27;
                  v72 = 1;
                  *((_DWORD *)v42 + 2) = 0;
                  v73 = 0;
                  goto LABEL_88;
                }
                v120 = a1;
              }
              sub_1BFE340(a1 + 8, v36);
              sub_1BFD9C0(v86, &v124, &v125);
              a1 = v120;
              v42 = v125;
              v27 = v124;
              v87 = *(_DWORD *)(v120 + 24) + 1;
              goto LABEL_114;
            }
            v119 = a1;
            sub_14672C0(a3, v26);
            v98 = *(_DWORD *)(a3 + 24);
            if ( v98 )
            {
              v99 = v98 - 1;
              v100 = *(_QWORD *)(a3 + 8);
              v101 = (v98 - 1) & v30;
              v102 = 0;
              v103 = 1;
              a1 = v119;
              v71 = *(_DWORD *)(a3 + 16) + 1;
              v32 = (__int64 *)(v100 + 16LL * v101);
              v104 = *v32;
              if ( v27 != *v32 )
              {
                while ( v104 != -8 )
                {
                  if ( !v102 && v104 == -16 )
                    v102 = v32;
                  v101 = v99 & (v103 + v101);
                  v32 = (__int64 *)(v100 + 16LL * v101);
                  v104 = *v32;
                  if ( v27 == *v32 )
                    goto LABEL_84;
                  ++v103;
                }
                if ( v102 )
                  v32 = v102;
              }
              goto LABEL_84;
            }
          }
          ++*(_DWORD *)(a3 + 16);
          BUG();
        }
      }
    }
LABEL_61:
    LODWORD(v125) = 1;
    v61 = v127;
    if ( v127 == v128 )
    {
      v113 = a1;
      sub_C88AB0((__int64)&v126, v127, &v125);
      a1 = v113;
    }
    else
    {
      if ( v127 )
      {
        *(_DWORD *)v127 = 1;
        v61 = v127;
      }
      v127 = v61 + 4;
    }
    do
    {
LABEL_40:
      if ( ++v12 == v18 )
        goto LABEL_44;
    }
    while ( *v12 == -8 || *v12 == -16 );
  }
  while ( v12 != v18 );
LABEL_44:
  v46 = v126;
  v47 = (v127 - (_BYTE *)v126) >> 2;
  if ( !v47 )
  {
LABEL_13:
    v19 = *(_DWORD *)(a3 + 24);
    if ( !v19 )
      goto LABEL_56;
    goto LABEL_14;
  }
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51 = 1;
  for ( j = *v126; v51 != v47; j = v126[v49] )
  {
    v53 = v51;
    do
    {
      v54 = &v46[v53];
      v55 = *v54;
      if ( *v54 > j )
      {
        *v54 = j;
        v46 = v126;
        j = v55;
      }
      ++v53;
    }
    while ( v53 != v47 );
    v46[v49] = j;
    v56 = v48 + j;
    v46 = v126;
    v48 = v51++;
    if ( v50 < v56 )
      v50 = v56;
    ++v49;
  }
  v57 = v48 + j;
  v19 = *(_DWORD *)(a3 + 24);
  if ( v57 < v50 )
    v57 = v50;
  v122 += v57;
  if ( !v19 )
  {
LABEL_56:
    ++*(_QWORD *)a3;
    goto LABEL_57;
  }
LABEL_14:
  v20 = v123;
  v21 = *(_QWORD *)(a3 + 8);
  v22 = (v19 - 1) & (((unsigned int)v123 >> 9) ^ ((unsigned int)v123 >> 4));
  v23 = (__int64 *)(v21 + 16LL * v22);
  v24 = *v23;
  if ( *v23 != v123 )
  {
    v88 = 1;
    v89 = 0;
    while ( v24 != -8 )
    {
      if ( !v89 && v24 == -16 )
        v89 = v23;
      v22 = (v19 - 1) & (v88 + v22);
      v23 = (__int64 *)(v21 + 16LL * v22);
      v24 = *v23;
      if ( v123 == *v23 )
        goto LABEL_15;
      ++v88;
    }
    v90 = *(_DWORD *)(a3 + 16);
    if ( v89 )
      v23 = v89;
    ++*(_QWORD *)a3;
    v58 = v90 + 1;
    if ( 4 * (v90 + 1) >= 3 * v19 )
    {
LABEL_57:
      v19 *= 2;
    }
    else if ( v19 - *(_DWORD *)(a3 + 20) - v58 > v19 >> 3 )
    {
      goto LABEL_124;
    }
    sub_14672C0(a3, v19);
    sub_1463AD0(a3, &v123, &v125);
    v23 = v125;
    v20 = v123;
    v58 = *(_DWORD *)(a3 + 16) + 1;
LABEL_124:
    *(_DWORD *)(a3 + 16) = v58;
    if ( *v23 != -8 )
      --*(_DWORD *)(a3 + 20);
    *v23 = v20;
    *((_DWORD *)v23 + 2) = 0;
  }
LABEL_15:
  *((_DWORD *)v23 + 2) = v122;
  if ( v126 )
    j_j___libc_free_0(v126, v128 - (_BYTE *)v126);
  j___libc_free_0(v130);
  return v122;
}
