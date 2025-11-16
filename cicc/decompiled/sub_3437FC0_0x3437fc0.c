// Function: sub_3437FC0
// Address: 0x3437fc0
//
char __fastcall sub_3437FC0(__int64 a1, __int64 a2)
{
  unsigned __int64 v3; // rdx
  __int64 v4; // r8
  __int64 v5; // r9
  __int64 v6; // r15
  unsigned __int64 v7; // rbx
  unsigned __int64 v8; // r13
  int v9; // r14d
  __int64 v10; // rax
  __int64 v11; // rcx
  unsigned __int64 *v12; // rsi
  __int64 v13; // rdi
  __int64 v14; // rdx
  __int64 v15; // rdx
  unsigned int v16; // eax
  __int64 v17; // r15
  unsigned int v18; // esi
  int v19; // ecx
  __int64 v20; // r9
  __int64 v21; // rax
  unsigned int v22; // r11d
  __int64 v23; // rdi
  unsigned int v24; // r11d
  unsigned int v25; // esi
  __int64 v26; // rdi
  __int64 v27; // r11
  unsigned int i; // r9d
  __int64 v29; // rdx
  __int64 v30; // r8
  unsigned int v31; // r9d
  __int64 *v32; // rcx
  __int64 v33; // rax
  __int16 v34; // dx
  __int64 v35; // rax
  __int64 v36; // rdx
  __int64 v37; // rdx
  _DWORD *v38; // rax
  int v39; // edx
  __int64 v40; // r12
  int v41; // esi
  int v42; // r8d
  __int64 v43; // r9
  __int64 v44; // r10
  int v45; // r11d
  unsigned int v46; // edi
  __int64 v47; // rax
  unsigned int v48; // edi
  int v49; // eax
  int v50; // esi
  __int64 v51; // rdi
  __int64 v52; // r10
  int v53; // r9d
  unsigned int n; // edx
  unsigned int v55; // edx
  int v56; // eax
  int v57; // edx
  __int64 v58; // rcx
  __int64 v59; // r9
  unsigned int j; // eax
  int v61; // eax
  int v62; // edx
  int v63; // eax
  int v64; // esi
  __int64 v65; // r8
  __int64 v66; // rdi
  int v67; // r10d
  unsigned int ii; // ecx
  unsigned int v69; // ecx
  int v70; // eax
  int v71; // r9d
  int v72; // edi
  int v73; // eax
  int v74; // esi
  __int64 v75; // r9
  int v76; // r10d
  __int64 v77; // r8
  unsigned int m; // edx
  unsigned int v79; // edx
  int v80; // esi
  int v81; // eax
  __int64 v82; // rax
  unsigned __int64 *v83; // rax
  int v84; // eax
  int v85; // eax
  __int64 v86; // rdx
  unsigned int k; // ecx
  int v88; // ecx
  int v89; // r8d
  int v90; // edi
  int v91; // eax
  int v92; // esi
  __int64 v93; // rcx
  int v94; // r9d
  __int64 v95; // r8
  unsigned int jj; // r15d
  unsigned int v97; // r15d
  int v98; // r8d
  int v99; // edi
  int v100; // edi
  int v101; // esi
  int v102; // r9d
  int v103; // esi
  int v105; // [rsp+10h] [rbp-60h]
  int v106; // [rsp+18h] [rbp-58h]
  int v107; // [rsp+18h] [rbp-58h]
  unsigned int v108; // [rsp+18h] [rbp-58h]
  __int64 *v109; // [rsp+18h] [rbp-58h]
  __int64 v110; // [rsp+18h] [rbp-58h]
  int v111; // [rsp+18h] [rbp-58h]
  int v112; // [rsp+18h] [rbp-58h]
  int v113; // [rsp+18h] [rbp-58h]
  int v114; // [rsp+18h] [rbp-58h]
  unsigned __int64 v115[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v116; // [rsp+30h] [rbp-40h] BYREF
  __int64 v117; // [rsp+38h] [rbp-38h]

  v10 = sub_338B750(*(_QWORD *)a1, a2);
  v6 = *(_QWORD *)(a1 + 8);
  v7 = v10;
  v8 = v3;
  v9 = v3;
  LODWORD(v10) = *(_DWORD *)(v6 + 16);
  if ( !(_DWORD)v10 )
  {
    v11 = *(unsigned int *)(v6 + 40);
    v10 = *(_QWORD *)(v6 + 32);
    v12 = (unsigned __int64 *)(v10 + 16 * v11);
    v13 = (16 * v11) >> 4;
    v14 = (16 * v11) >> 6;
    if ( v14 )
    {
      v15 = v10 + (v14 << 6);
      while ( *(_QWORD *)v10 != v7 || v9 != *(_DWORD *)(v10 + 8) )
      {
        if ( *(_QWORD *)(v10 + 16) == v7 && v9 == *(_DWORD *)(v10 + 24) )
        {
          v10 += 16;
          if ( v12 != (unsigned __int64 *)v10 )
            return v10;
          goto LABEL_15;
        }
        if ( *(_QWORD *)(v10 + 32) == v7 && v9 == *(_DWORD *)(v10 + 40) )
        {
          v10 += 32;
          if ( v12 != (unsigned __int64 *)v10 )
            return v10;
          goto LABEL_15;
        }
        if ( *(_QWORD *)(v10 + 48) == v7 && v9 == *(_DWORD *)(v10 + 56) )
        {
          v10 += 48;
          if ( v12 != (unsigned __int64 *)v10 )
            return v10;
          goto LABEL_15;
        }
        v10 += 64;
        if ( v15 == v10 )
        {
          v13 = ((__int64)v12 - v10) >> 4;
          goto LABEL_10;
        }
      }
      goto LABEL_25;
    }
LABEL_10:
    if ( v13 != 2 )
    {
      if ( v13 != 3 )
      {
        if ( v13 != 1 )
        {
LABEL_15:
          if ( v11 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 44) )
          {
            sub_C8D5F0(v6 + 32, (const void *)(v6 + 48), v11 + 1, 0x10u, v4, v5);
            v12 = (unsigned __int64 *)(*(_QWORD *)(v6 + 32) + 16LL * *(unsigned int *)(v6 + 40));
          }
          *v12 = v7;
          v12[1] = v8;
          v16 = *(_DWORD *)(v6 + 40) + 1;
          *(_DWORD *)(v6 + 40) = v16;
          if ( v16 > 0x10 )
            sub_3437D00(v6);
          goto LABEL_19;
        }
LABEL_13:
        if ( *(_QWORD *)v10 != v7 || (_DWORD)v8 != *(_DWORD *)(v10 + 8) )
          goto LABEL_15;
LABEL_25:
        if ( v12 != (unsigned __int64 *)v10 )
          return v10;
        goto LABEL_15;
      }
      if ( *(_QWORD *)v10 == v7 && (_DWORD)v8 == *(_DWORD *)(v10 + 8) )
        goto LABEL_25;
      v10 += 16;
    }
    if ( *(_QWORD *)v10 == v7 && (_DWORD)v8 == *(_DWORD *)(v10 + 8) )
      goto LABEL_25;
    v10 += 16;
    goto LABEL_13;
  }
  v25 = *(_DWORD *)(v6 + 24);
  if ( !v25 )
  {
    ++*(_QWORD *)v6;
    goto LABEL_77;
  }
  v107 = 1;
  v26 = 0;
  v105 = *(_DWORD *)(v6 + 16);
  v27 = *(_QWORD *)(v6 + 8);
  for ( i = (v25 - 1) & (v3 + ((v7 >> 9) ^ (v7 >> 4))); ; i = (v25 - 1) & v31 )
  {
    v29 = v27 + 16LL * i;
    v30 = *(_QWORD *)v29;
    if ( *(_QWORD *)v29 == v7 && v9 == *(_DWORD *)(v29 + 8) )
      return v10;
    if ( !v30 )
      break;
LABEL_40:
    v31 = v107 + i;
    LOBYTE(v10) = ++v107;
  }
  v70 = *(_DWORD *)(v29 + 8);
  if ( v70 != -1 )
  {
    if ( !v26 && v70 == -2 )
      v26 = v27 + 16LL * i;
    goto LABEL_40;
  }
  if ( !v26 )
    v26 = v27 + 16LL * i;
  ++*(_QWORD *)v6;
  v81 = v105 + 1;
  if ( 4 * (v105 + 1) >= 3 * v25 )
  {
LABEL_77:
    sub_3437AD0(v6, 2 * v25);
    v56 = *(_DWORD *)(v6 + 24);
    if ( !v56 )
      goto LABEL_209;
    v57 = v56 - 1;
    v59 = 1;
    v30 = 0;
    for ( j = (v56 - 1) & (v8 + ((v7 >> 9) ^ (v7 >> 4))); ; j = v57 & v61 )
    {
      v58 = *(_QWORD *)(v6 + 8);
      v26 = v58 + 16LL * j;
      if ( *(_QWORD *)v26 == v7 && v9 == *(_DWORD *)(v26 + 8) )
        break;
      if ( !*(_QWORD *)v26 )
      {
        v103 = *(_DWORD *)(v26 + 8);
        if ( v103 == -1 )
        {
LABEL_199:
          v81 = *(_DWORD *)(v6 + 16) + 1;
          if ( v30 )
            v26 = v30;
          goto LABEL_127;
        }
        if ( v103 == -2 && !v30 )
          v30 = v58 + 16LL * j;
      }
      v61 = v59 + j;
      v59 = (unsigned int)(v59 + 1);
    }
LABEL_126:
    v81 = *(_DWORD *)(v6 + 16) + 1;
    goto LABEL_127;
  }
  v59 = v25 >> 3;
  if ( v25 - *(_DWORD *)(v6 + 20) - v81 <= (unsigned int)v59 )
  {
    sub_3437AD0(v6, v25);
    v84 = *(_DWORD *)(v6 + 24);
    if ( v84 )
    {
      v85 = v84 - 1;
      v59 = 1;
      v30 = 0;
      for ( k = v85 & (v8 + ((v7 >> 9) ^ (v7 >> 4))); ; k = v85 & v88 )
      {
        v86 = *(_QWORD *)(v6 + 8);
        v26 = v86 + 16LL * k;
        if ( *(_QWORD *)v26 == v7 && v9 == *(_DWORD *)(v26 + 8) )
          break;
        if ( !*(_QWORD *)v26 )
        {
          v101 = *(_DWORD *)(v26 + 8);
          if ( v101 == -1 )
            goto LABEL_199;
          if ( !v30 && v101 == -2 )
            v30 = v86 + 16LL * k;
        }
        v88 = v59 + k;
        v59 = (unsigned int)(v59 + 1);
      }
      goto LABEL_126;
    }
LABEL_209:
    ++*(_DWORD *)(v6 + 16);
    BUG();
  }
LABEL_127:
  *(_DWORD *)(v6 + 16) = v81;
  if ( *(_QWORD *)v26 || *(_DWORD *)(v26 + 8) != -1 )
    --*(_DWORD *)(v6 + 20);
  *(_QWORD *)v26 = v7;
  *(_DWORD *)(v26 + 8) = v8;
  v82 = *(unsigned int *)(v6 + 40);
  if ( v82 + 1 > (unsigned __int64)*(unsigned int *)(v6 + 44) )
  {
    sub_C8D5F0(v6 + 32, (const void *)(v6 + 48), v82 + 1, 0x10u, v30, v59);
    v82 = *(unsigned int *)(v6 + 40);
  }
  v83 = (unsigned __int64 *)(*(_QWORD *)(v6 + 32) + 16 * v82);
  *v83 = v7;
  v83[1] = v8;
  ++*(_DWORD *)(v6 + 40);
LABEL_19:
  v17 = *(_QWORD *)(a1 + 16);
  v18 = *(_DWORD *)(v17 + 24);
  v19 = *(_DWORD *)(*(_QWORD *)(a1 + 8) + 40LL) - 1;
  if ( !v18 )
  {
    ++*(_QWORD *)v17;
    goto LABEL_67;
  }
  v20 = *(_QWORD *)(v17 + 8);
  v106 = 1;
  v21 = 0;
  v22 = (v18 - 1) & (v8 + ((v7 >> 9) ^ (v7 >> 4)));
  while ( 2 )
  {
    v23 = v20 + 24LL * v22;
    if ( *(_QWORD *)v23 == v7 && v9 == *(_DWORD *)(v23 + 8) )
    {
      v10 = v23 + 16;
      goto LABEL_45;
    }
    if ( *(_QWORD *)v23 )
    {
LABEL_23:
      v24 = v106 + v22;
      ++v106;
      v22 = (v18 - 1) & v24;
      continue;
    }
    break;
  }
  v62 = *(_DWORD *)(v23 + 8);
  if ( v62 != -1 )
  {
    if ( !v21 && v62 == -2 )
      v21 = v20 + 24LL * v22;
    goto LABEL_23;
  }
  if ( !v21 )
    v21 = v20 + 24LL * v22;
  v72 = *(_DWORD *)(v17 + 16);
  ++*(_QWORD *)v17;
  v71 = v72 + 1;
  if ( 4 * (v72 + 1) < 3 * v18 )
  {
    if ( v18 - *(_DWORD *)(v17 + 20) - v71 <= v18 >> 3 )
    {
      v113 = v19;
      sub_3437080(v17, v18);
      v73 = *(_DWORD *)(v17 + 24);
      if ( v73 )
      {
        v74 = v73 - 1;
        v75 = *(_QWORD *)(v17 + 8);
        v76 = 1;
        v19 = v113;
        v77 = 0;
        for ( m = (v73 - 1) & (v8 + ((v7 >> 9) ^ (v7 >> 4))); ; m = v74 & v79 )
        {
          v21 = v75 + 24LL * m;
          if ( *(_QWORD *)v21 == v7 && v9 == *(_DWORD *)(v21 + 8) )
            break;
          if ( !*(_QWORD *)v21 )
          {
            v99 = *(_DWORD *)(v21 + 8);
            if ( v99 == -1 )
            {
              if ( v77 )
                v21 = v77;
              v71 = *(_DWORD *)(v17 + 16) + 1;
              goto LABEL_102;
            }
            if ( !v77 && v99 == -2 )
              v77 = v75 + 24LL * m;
          }
          v79 = v76 + m;
          ++v76;
        }
        goto LABEL_101;
      }
LABEL_208:
      ++*(_DWORD *)(v17 + 16);
      BUG();
    }
    goto LABEL_102;
  }
LABEL_67:
  v111 = v19;
  sub_3437080(v17, 2 * v18);
  v49 = *(_DWORD *)(v17 + 24);
  if ( !v49 )
    goto LABEL_208;
  v50 = v49 - 1;
  v19 = v111;
  v52 = 0;
  v53 = 1;
  for ( n = (v49 - 1) & (v8 + ((v7 >> 9) ^ (v7 >> 4))); ; n = v50 & v55 )
  {
    v51 = *(_QWORD *)(v17 + 8);
    v21 = v51 + 24LL * n;
    if ( *(_QWORD *)v21 == v7 && v9 == *(_DWORD *)(v21 + 8) )
      break;
    if ( !*(_QWORD *)v21 )
    {
      v98 = *(_DWORD *)(v21 + 8);
      if ( v98 == -1 )
      {
        if ( v52 )
          v21 = v52;
        v71 = *(_DWORD *)(v17 + 16) + 1;
        goto LABEL_102;
      }
      if ( v98 == -2 && !v52 )
        v52 = v51 + 24LL * n;
    }
    v55 = v53 + n;
    ++v53;
  }
LABEL_101:
  v71 = *(_DWORD *)(v17 + 16) + 1;
LABEL_102:
  *(_DWORD *)(v17 + 16) = v71;
  if ( *(_QWORD *)v21 || *(_DWORD *)(v21 + 8) != -1 )
    --*(_DWORD *)(v17 + 20);
  *(_QWORD *)v21 = v7;
  v10 = v21 + 16;
  *(_DWORD *)(v10 - 8) = v8;
  *(_DWORD *)v10 = 0;
LABEL_45:
  *(_DWORD *)v10 = v19;
  LODWORD(v10) = *(_DWORD *)(*(_QWORD *)(a1 + 24) + 16LL);
  if ( **(_DWORD **)(a1 + 32) == (_DWORD)v10 )
    return v10;
  v115[0] = v7;
  v32 = *(__int64 **)(a1 + 40);
  v33 = *(_QWORD *)(v7 + 48) + 16LL * (unsigned int)v8;
  v115[1] = v8;
  v34 = *(_WORD *)v33;
  v35 = *(_QWORD *)(v33 + 8);
  v116 = v34;
  v117 = v35;
  if ( v34 )
  {
    LOBYTE(v10) = (unsigned __int16)(v34 - 17) <= 0xD3u;
  }
  else
  {
    v109 = v32;
    LOBYTE(v10) = sub_30070B0((__int64)&v116);
    v32 = v109;
  }
  if ( (_BYTE)v10 )
    return v10;
  v36 = *v32;
  if ( *(_QWORD *)(*v32 + 184) )
  {
    v110 = *v32;
    v10 = sub_3434600(v36 + 144, v115);
    v37 = v110 + 152;
  }
  else
  {
    v10 = *(_QWORD *)v36;
    v37 = *(_QWORD *)v36 + 16LL * *(unsigned int *)(v36 + 8);
    if ( v10 == v37 )
      goto LABEL_56;
    while ( v7 != *(_QWORD *)v10 || (_DWORD)v8 != *(_DWORD *)(v10 + 8) )
    {
      v10 += 16;
      if ( v37 == v10 )
        goto LABEL_56;
    }
  }
  if ( v37 != v10 )
    return v10;
LABEL_56:
  LOBYTE(v10) = sub_3433840(v7, v8, v37);
  if ( (_BYTE)v10 )
    return v10;
  v38 = *(_DWORD **)(a1 + 48);
  v39 = (*v38)++;
  v40 = *(_QWORD *)(a1 + 24);
  v41 = *(_DWORD *)(v40 + 24);
  if ( !v41 )
  {
    ++*(_QWORD *)v40;
    goto LABEL_88;
  }
  v108 = *(_DWORD *)(v40 + 24);
  v42 = v41 - 1;
  v43 = *(_QWORD *)(v40 + 8);
  v44 = 0;
  v45 = 1;
  v46 = (v41 - 1) & (v8 + ((v7 >> 9) ^ (v7 >> 4)));
  while ( 2 )
  {
    v47 = v43 + 24LL * v46;
    if ( v7 == *(_QWORD *)v47 && v9 == *(_DWORD *)(v47 + 8) )
      goto LABEL_65;
    if ( *(_QWORD *)v47 )
    {
LABEL_61:
      v48 = v45 + v46;
      ++v45;
      v46 = v42 & v48;
      continue;
    }
    break;
  }
  v80 = *(_DWORD *)(v47 + 8);
  if ( v80 != -1 )
  {
    if ( !v44 && v80 == -2 )
      v44 = v43 + 24LL * v46;
    goto LABEL_61;
  }
  v90 = *(_DWORD *)(v40 + 16);
  v41 = *(_DWORD *)(v40 + 24);
  if ( v44 )
    v47 = v44;
  ++*(_QWORD *)v40;
  v89 = v90 + 1;
  if ( 4 * (v90 + 1) >= 3 * v108 )
  {
LABEL_88:
    v112 = v39;
    sub_34372C0(v40, 2 * v41);
    v63 = *(_DWORD *)(v40 + 24);
    if ( !v63 )
      goto LABEL_210;
    v64 = v63 - 1;
    v39 = v112;
    v66 = 0;
    v67 = 1;
    for ( ii = (v63 - 1) & (v8 + ((v7 >> 9) ^ (v7 >> 4))); ; ii = v64 & v69 )
    {
      v65 = *(_QWORD *)(v40 + 8);
      v47 = v65 + 24LL * ii;
      if ( v7 == *(_QWORD *)v47 && v9 == *(_DWORD *)(v47 + 8) )
        break;
      if ( !*(_QWORD *)v47 )
      {
        v102 = *(_DWORD *)(v47 + 8);
        if ( v102 == -1 )
        {
          if ( v66 )
            v47 = v66;
          v89 = *(_DWORD *)(v40 + 16) + 1;
          goto LABEL_147;
        }
        if ( v102 == -2 && !v66 )
          v66 = v65 + 24LL * ii;
      }
      v69 = v67 + ii;
      ++v67;
    }
LABEL_146:
    v89 = *(_DWORD *)(v40 + 16) + 1;
    goto LABEL_147;
  }
  if ( v108 - *(_DWORD *)(v40 + 20) - v89 <= v108 >> 3 )
  {
    v114 = v39;
    sub_34372C0(v40, v41);
    v91 = *(_DWORD *)(v40 + 24);
    if ( v91 )
    {
      v92 = v91 - 1;
      v39 = v114;
      v93 = 0;
      v94 = 1;
      for ( jj = (v91 - 1) & (v8 + ((v7 >> 9) ^ (v7 >> 4))); ; jj = v92 & v97 )
      {
        v95 = *(_QWORD *)(v40 + 8);
        v47 = v95 + 24LL * jj;
        if ( v7 == *(_QWORD *)v47 && v9 == *(_DWORD *)(v47 + 8) )
          break;
        if ( !*(_QWORD *)v47 )
        {
          v100 = *(_DWORD *)(v47 + 8);
          if ( v100 == -1 )
          {
            if ( v93 )
              v47 = v93;
            v89 = *(_DWORD *)(v40 + 16) + 1;
            goto LABEL_147;
          }
          if ( !v93 && v100 == -2 )
            v93 = v95 + 24LL * jj;
        }
        v97 = v94 + jj;
        ++v94;
      }
      goto LABEL_146;
    }
LABEL_210:
    ++*(_DWORD *)(v40 + 16);
    BUG();
  }
LABEL_147:
  *(_DWORD *)(v40 + 16) = v89;
  if ( *(_QWORD *)v47 || *(_DWORD *)(v47 + 8) != -1 )
    --*(_DWORD *)(v40 + 20);
  *(_QWORD *)v47 = v7;
  *(_DWORD *)(v47 + 8) = v8;
  *(_DWORD *)(v47 + 16) = 0;
LABEL_65:
  v10 = v47 + 16;
  *(_DWORD *)v10 = v39;
  return v10;
}
