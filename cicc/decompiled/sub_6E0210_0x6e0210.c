// Function: sub_6E0210
// Address: 0x6e0210
//
__int64 __fastcall sub_6E0210(__int64 *a1, __int64 *a2)
{
  int v3; // r8d
  __int64 v4; // rsi
  int v5; // edx
  __int64 v6; // rcx
  char v7; // al
  int v8; // r8d
  __int64 v9; // rsi
  int i; // edx
  __int64 v11; // rcx
  char v12; // al
  __int64 *v13; // r15
  unsigned __int64 *v14; // rax
  unsigned __int64 *v15; // r14
  unsigned __int64 *v16; // rdx
  int v17; // r13d
  __int64 v18; // rdi
  signed int v19; // ebx
  unsigned int v20; // r15d
  __int64 v21; // rcx
  __int64 v22; // rdi
  unsigned __int64 v23; // r10
  unsigned int v24; // eax
  __int64 v25; // rsi
  unsigned __int64 *v26; // rdx
  unsigned __int64 v27; // r11
  unsigned __int64 *v28; // rdx
  unsigned __int64 *v29; // rax
  int v30; // edx
  __int64 v31; // rax
  __int64 v32; // rsi
  signed int v33; // r10d
  signed int v34; // r11d
  signed int v35; // eax
  __int64 v36; // rcx
  _DWORD *v37; // rdx
  char v38; // al
  __int64 *v39; // r12
  unsigned __int64 *v40; // rbx
  __int64 v41; // rax
  int v42; // r13d
  signed int v43; // r14d
  __int64 v44; // rdx
  __int64 v45; // r8
  __int64 m; // rcx
  unsigned int v47; // ecx
  unsigned __int64 *v48; // rax
  int v49; // ecx
  int *v50; // rax
  signed int v51; // esi
  _DWORD *v52; // rdx
  __int64 v53; // rcx
  char v54; // al
  signed int v55; // r8d
  signed int v56; // esi
  int v57; // r14d
  __int64 v58; // rax
  __int64 v59; // rdx
  __int64 v60; // r13
  __int64 v61; // rax
  __int64 v62; // rbx
  __int64 v63; // rax
  __int64 v64; // rsi
  unsigned __int64 *v65; // r15
  unsigned int v66; // ebx
  unsigned __int64 *v68; // rax
  unsigned int v69; // r10d
  unsigned __int64 *v70; // r11
  unsigned __int64 *v71; // rdx
  unsigned __int64 *v72; // rsi
  __int64 v73; // r8
  unsigned __int64 v74; // rdi
  unsigned __int64 j; // rdx
  unsigned int v76; // edx
  unsigned __int64 *v77; // rax
  int v78; // edx
  __int64 v79; // rdi
  int *v80; // rax
  int *v81; // r8
  int *v82; // rsi
  int *v83; // r9
  signed int v84; // r10d
  unsigned __int64 *v85; // rax
  unsigned int v86; // r11d
  unsigned __int64 *v87; // r10
  unsigned __int64 *v88; // rdx
  unsigned __int64 *v89; // rsi
  __int64 v90; // r8
  unsigned __int64 v91; // rdi
  unsigned __int64 k; // rdx
  unsigned int v93; // edx
  unsigned __int64 *v94; // rax
  int v95; // edx
  unsigned int v96; // eax
  unsigned int v97; // eax
  __int64 v98; // rsi
  int *v100; // [rsp+10h] [rbp-70h]
  int v101; // [rsp+18h] [rbp-68h]
  int *v102; // [rsp+18h] [rbp-68h]
  unsigned int v103; // [rsp+20h] [rbp-60h]
  int v104; // [rsp+20h] [rbp-60h]
  unsigned __int64 *v105; // [rsp+28h] [rbp-58h]
  __int64 v106; // [rsp+28h] [rbp-58h]
  unsigned int v107; // [rsp+30h] [rbp-50h]
  unsigned __int64 *v108; // [rsp+30h] [rbp-50h]
  unsigned __int64 *v109; // [rsp+30h] [rbp-50h]
  __int64 v110; // [rsp+38h] [rbp-48h]
  __int64 *v111; // [rsp+40h] [rbp-40h]
  int *v112; // [rsp+40h] [rbp-40h]
  int v113; // [rsp+48h] [rbp-38h]
  __int64 v114; // [rsp+48h] [rbp-38h]
  int *v115; // [rsp+48h] [rbp-38h]

  if ( (int)a1[2] > 0 )
  {
    v3 = a1[2];
    v4 = 0;
    v5 = 0;
    do
    {
      while ( 1 )
      {
        v6 = v4 + *a1;
        v7 = *(_BYTE *)v6 & 3;
        if ( v7 != 3 )
          break;
        ++v5;
        *(_DWORD *)(v6 + 8) = 0;
        v4 += 24;
        if ( v3 == v5 )
          goto LABEL_8;
      }
      if ( !v7 )
        *(_DWORD *)(v6 + 8) = v5;
      ++v5;
      v4 += 24;
    }
    while ( v3 != v5 );
  }
LABEL_8:
  v8 = a2[2];
  if ( v8 > 0 )
  {
    v9 = 0;
    for ( i = 0; i != v8; ++i )
    {
      while ( 1 )
      {
        v11 = v9 + *a2;
        v12 = *(_BYTE *)v11 & 3;
        if ( v12 != 2 )
          break;
        ++i;
        *(_DWORD *)(v11 + 8) = 0;
        v9 += 24;
        if ( v8 == i )
          goto LABEL_15;
      }
      if ( !v12 )
        *(_DWORD *)(v11 + 8) = i;
      v9 += 24;
    }
  }
LABEL_15:
  v13 = a2;
  while ( 2 )
  {
    v14 = (unsigned __int64 *)sub_823970(128);
    v15 = v14;
    v16 = v14 + 16;
    do
    {
      if ( v14 )
        *v14 = 0;
      v14 += 2;
    }
    while ( v16 != v14 );
    v17 = 1;
    v18 = *a1;
    v113 = 0;
    v111 = v13;
    v19 = sub_6DFA00(*a1, 0);
    v20 = 7;
    while ( 2 )
    {
      v21 = 24LL * v19;
      v22 = v21 + v18;
      v23 = *(_QWORD *)(v22 + 16);
      v24 = v20 & (v23 >> 3);
      v25 = v24;
      v26 = &v15[2 * v24];
      v27 = *v26;
      if ( *v26 )
      {
        do
        {
          if ( v23 == v27 )
          {
            v29 = &v15[2 * v25];
            v30 = *((_DWORD *)v29 + 2);
            *((_DWORD *)v29 + 2) = v19;
            if ( v30 )
            {
              v31 = *a1 + 24LL * v30;
              *(_DWORD *)(v22 + 8) = *(_DWORD *)(v31 + 8);
              *(_DWORD *)(v31 + 8) = v19;
            }
            goto LABEL_27;
          }
          v24 = v20 & (v24 + 1);
          v25 = v24;
          v28 = &v15[2 * v24];
          v27 = *v28;
        }
        while ( *v28 );
        *v28 = v23;
        if ( v23 )
          *((_DWORD *)v28 + 2) = v19;
        ++v113;
        if ( 2 * v113 > v20 )
        {
          v68 = (unsigned __int64 *)sub_823970(16LL * (2 * v20 + 2));
          v69 = 2 * v20 + 1;
          v70 = v68;
          if ( 2 * v20 != -2 )
          {
            v71 = &v68[2 * v69 + 2];
            do
            {
              if ( v68 )
                *v68 = 0;
              v68 += 2;
            }
            while ( v71 != v68 );
          }
          if ( v20 != -1 )
          {
            v72 = v15;
            v73 = (__int64)&v15[2 * v20 + 2];
            do
            {
              while ( 1 )
              {
                v74 = *v72;
                if ( *v72 )
                  break;
                v72 += 2;
                if ( (unsigned __int64 *)v73 == v72 )
                  goto LABEL_92;
              }
              for ( j = v74 >> 3; ; LODWORD(j) = v76 + 1 )
              {
                v76 = v69 & j;
                v77 = &v70[2 * v76];
                if ( !*v77 )
                  break;
              }
              *v77 = v74;
              v78 = *((_DWORD *)v72 + 2);
              v72 += 2;
              *((_DWORD *)v77 + 2) = v78;
            }
            while ( (unsigned __int64 *)v73 != v72 );
          }
LABEL_92:
          v108 = v70;
          sub_823A00(v15, 16LL * (v20 + 1));
          v21 = 24LL * v19;
          v20 = 2 * v20 + 1;
          v15 = v108;
        }
      }
      else
      {
        *v26 = v23;
        if ( v23 )
          *((_DWORD *)v26 + 2) = v19;
        ++v113;
        if ( 2 * v113 > v20 )
        {
          v85 = (unsigned __int64 *)sub_823970(16LL * (2 * v20 + 2));
          v86 = 2 * v20 + 1;
          v87 = v85;
          if ( 2 * v20 != -2 )
          {
            v88 = &v85[2 * v86 + 2];
            do
            {
              if ( v85 )
                *v85 = 0;
              v85 += 2;
            }
            while ( v88 != v85 );
          }
          if ( v20 != -1 )
          {
            v89 = v15;
            v90 = (__int64)&v15[2 * v20 + 2];
            do
            {
              while ( 1 )
              {
                v91 = *v89;
                if ( *v89 )
                  break;
                v89 += 2;
                if ( (unsigned __int64 *)v90 == v89 )
                  goto LABEL_123;
              }
              for ( k = v91 >> 3; ; LODWORD(k) = v93 + 1 )
              {
                v93 = v86 & k;
                v94 = &v87[2 * v93];
                if ( !*v94 )
                  break;
              }
              *v94 = v91;
              v95 = *((_DWORD *)v89 + 2);
              v89 += 2;
              *((_DWORD *)v94 + 2) = v95;
            }
            while ( (unsigned __int64 *)v90 != v89 );
          }
LABEL_123:
          v109 = v87;
          sub_823A00(v15, 16LL * (v20 + 1));
          v21 = 24LL * v19;
          v20 = 2 * v20 + 1;
          v15 = v109;
        }
      }
LABEL_27:
      v32 = *a1;
      v33 = *(_DWORD *)(*a1 + v21 + 4);
      v18 = *a1;
      if ( v33 != -1 )
      {
        while ( 1 )
        {
          v36 = 24LL * v33;
          v37 = (_DWORD *)(v32 + v36);
          v38 = *(_BYTE *)(v32 + v36) & 3;
          if ( v38 == 2 )
            break;
          if ( v38 != 3 )
LABEL_125:
            sub_721090(v18);
          if ( v37[2] || !v17 )
            goto LABEL_30;
          v37[2] = 1;
          v32 = *a1;
          v17 = 0;
          v35 = *(_DWORD *)(*a1 + v36 + 4);
          if ( v35 == -1 )
          {
LABEL_37:
            v107 = v17;
            v39 = v111;
            v40 = v15;
            goto LABEL_38;
          }
LABEL_31:
          v19 = v33;
          v18 = v32;
          v33 = v35;
        }
        v34 = *v37 >> 2;
        if ( v34 > v19 )
        {
          if ( v17 )
          {
            v84 = v33 + 1;
            if ( v34 > v84 )
            {
              do
                ++v84;
              while ( v34 != v84 );
              v18 = *a1;
            }
          }
          v19 = sub_6DFA00(v18, v34);
          if ( v19 == -1 )
            break;
          continue;
        }
LABEL_30:
        v35 = v37[1];
        if ( v35 == -1 )
          goto LABEL_37;
        goto LABEL_31;
      }
      break;
    }
    v107 = v17;
    v40 = v15;
    v39 = v111;
LABEL_38:
    while ( 2 )
    {
      v41 = sub_823970(80);
      v18 = *v39;
      v112 = (int *)v41;
      v42 = 1;
      v114 = 0;
      v110 = 10;
      v43 = sub_6DFA40(*v39, 0);
      while ( 2 )
      {
        v44 = 24LL * v43;
        v45 = v18 + v44;
        for ( m = *(_QWORD *)(v18 + v44 + 16) >> 3; ; LODWORD(m) = v47 + 1 )
        {
          v47 = v20 & m;
          v48 = &v40[2 * v47];
          if ( *(_QWORD *)(v18 + v44 + 16) == *v48 )
            break;
          if ( !*v48 )
            goto LABEL_48;
        }
        v49 = *((_DWORD *)v48 + 2);
        if ( v49 )
        {
          if ( v114 == v110 )
          {
            if ( v114 <= 1 )
            {
              v106 = 2;
              v79 = 16;
            }
            else
            {
              v106 = v114 + (v114 >> 1) + 1;
              v79 = 8 * v106;
            }
            v104 = *((_DWORD *)v48 + 2);
            v80 = (int *)sub_823970(v79);
            v81 = v80;
            if ( v114 > 0 )
            {
              v82 = v112;
              v83 = &v80[2 * v114];
              do
              {
                if ( v80 )
                  *(_QWORD *)v80 = *(_QWORD *)v82;
                v80 += 2;
                v82 += 2;
              }
              while ( v83 != (_QWORD *)v80 );
            }
            v102 = v81;
            sub_823A00(v112, 8 * v110);
            v49 = v104;
            v44 = 24LL * v43;
            v110 = v106;
            v112 = v102;
          }
          v50 = &v112[2 * v114];
          if ( v50 )
          {
            *v50 = v49;
            v50[1] = v43;
          }
          ++v114;
          v18 = *v39;
          v45 = *v39 + v44;
        }
LABEL_48:
        v51 = *(_DWORD *)(v45 + 4);
        if ( v51 == -1 )
          break;
        while ( 2 )
        {
          v53 = 24LL * v51;
          v52 = (_DWORD *)(v18 + v53);
          v54 = *(_BYTE *)(v18 + v53) & 3;
          if ( v54 == 2 )
          {
            if ( !v52[2] && v42 )
            {
              v52[2] = 1;
              v42 = 0;
              v52 = (_DWORD *)(v53 + *v39);
            }
LABEL_53:
            if ( v52[1] == -1 )
              goto LABEL_63;
            v18 = *v39;
            v43 = v51;
            v51 = v52[1];
            continue;
          }
          break;
        }
        if ( v54 != 3 )
          goto LABEL_125;
        v55 = *v52 >> 2;
        if ( v55 <= v43 )
          goto LABEL_53;
        if ( v42 )
        {
          v56 = v51 + 1;
          if ( v55 > v56 )
          {
            do
              ++v56;
            while ( v55 != v56 );
            v18 = *v39;
          }
        }
        v43 = sub_6DFA40(v18, v55);
        if ( v43 != -1 )
          continue;
        break;
      }
LABEL_63:
      if ( !v114 )
      {
        v97 = v20;
        v65 = v40;
        v66 = v97;
LABEL_74:
        sub_823A00(v112, 8 * v110);
        sub_823A00(v65, 16LL * (v66 + 1));
        return 0;
      }
      sub_865900(0);
      v100 = &v112[2 * v114];
      if ( v100 == v112 )
      {
        v96 = v20;
        v65 = v40;
        v66 = v96;
LABEL_73:
        sub_864110();
        goto LABEL_74;
      }
      v115 = v112;
      v105 = v40;
      v101 = v42;
      v103 = v20;
LABEL_66:
      v57 = *v115;
      v58 = sub_6DF4F0(v39, *(_DWORD *)(*v39 + 24LL * v115[1]) >> 2);
      v59 = *a1;
      v60 = v58;
      v61 = v57;
      while ( 1 )
      {
        v62 = 24 * v61;
        v63 = sub_6DF4F0(a1, *(_DWORD *)(v59 + 24 * v61) >> 2);
        if ( (unsigned int)sub_89AB40(v63, v60, 128) )
          break;
        v59 = *a1;
        v61 = *(int *)(*a1 + v62 + 8);
        if ( v57 == (_DWORD)v61 )
        {
          v115 += 2;
          if ( v100 == v115 )
          {
            v65 = v105;
            v66 = v103;
            goto LABEL_73;
          }
          goto LABEL_66;
        }
      }
      v40 = v105;
      sub_864110();
      v64 = 8 * v110;
      if ( !v101 )
      {
        sub_823A00(v112, v64);
        continue;
      }
      break;
    }
    v13 = v39;
    sub_823A00(v112, v64);
    v98 = 16LL * (v103 + 1);
    if ( !v107 )
    {
      sub_823A00(v105, v98);
      continue;
    }
    break;
  }
  sub_823A00(v105, v98);
  return v107;
}
