// Function: sub_16A6110
// Address: 0x16a6110
//
unsigned __int64 __fastcall sub_16A6110(
        __int64 *a1,
        unsigned int a2,
        __int64 *a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r14
  __int64 v8; // rbx
  int v9; // esi
  __int64 v10; // rcx
  size_t v11; // rdx
  _BYTE *v12; // r11
  _DWORD *v13; // r11
  int v14; // edx
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 *v18; // rdx
  int v19; // ecx
  __int64 v20; // rax
  __int64 v21; // r8
  _BYTE *v22; // r11
  __int64 v23; // rax
  unsigned int *v24; // rdx
  int v25; // ecx
  int v26; // eax
  _DWORD *i; // rax
  int v28; // r13d
  __int64 v29; // rax
  _DWORD *v30; // rbx
  __int64 v31; // r12
  unsigned int v32; // edx
  __int64 v33; // r10
  int v34; // r9d
  unsigned __int64 v35; // rcx
  unsigned __int64 v36; // rax
  unsigned __int64 v37; // rdx
  unsigned __int64 v38; // rtt
  __int64 v39; // rdi
  __int64 v40; // r8
  __int64 v41; // rbx
  __int64 v42; // rsi
  unsigned __int64 v43; // rdx
  int v44; // r12d
  unsigned int *v45; // rsi
  int v46; // r8d
  unsigned int v47; // ecx
  __int64 v48; // r15
  unsigned int *v49; // rbx
  __int64 v50; // rax
  __int64 v51; // rdx
  _DWORD *v52; // r13
  unsigned __int64 v53; // rdx
  __int64 v54; // rax
  int v55; // r14d
  unsigned int v56; // edx
  unsigned __int64 result; // rax
  __int64 v58; // rsi
  unsigned int v59; // edi
  int v60; // edi
  __int64 v61; // r9
  unsigned int v62; // edx
  __int64 v63; // rax
  __int64 v64; // rdi
  unsigned __int8 v65; // si
  unsigned int *v66; // rdi
  unsigned int v67; // r12d
  unsigned int v68; // eax
  unsigned int v69; // edi
  unsigned __int64 v70; // rcx
  __int64 v71; // r8
  unsigned int v72; // edx
  __int64 v73; // rax
  __int64 v74; // rdi
  unsigned int *v75; // rsi
  __int64 v76; // r9
  unsigned int v77; // edx
  int v78; // edi
  unsigned int *v79; // rsi
  unsigned int v80; // edi
  unsigned int v81; // edx
  unsigned int v82; // r14d
  __int64 v83; // rax
  __int64 v84; // rax
  int v85; // [rsp+4h] [rbp-2ACh]
  __int64 v86; // [rsp+8h] [rbp-2A8h]
  char v89; // [rsp+28h] [rbp-288h]
  _DWORD *v91; // [rsp+30h] [rbp-280h]
  unsigned int *v92; // [rsp+38h] [rbp-278h]
  unsigned int *v93; // [rsp+40h] [rbp-270h]
  int v95; // [rsp+4Ch] [rbp-264h]
  _BYTE *v96; // [rsp+50h] [rbp-260h]
  unsigned int *v97; // [rsp+58h] [rbp-258h]
  __int64 v98; // [rsp+58h] [rbp-258h]
  __int64 v99; // [rsp+58h] [rbp-258h]
  __int64 v100; // [rsp+60h] [rbp-250h]
  __int64 v101; // [rsp+60h] [rbp-250h]
  unsigned int v103; // [rsp+68h] [rbp-248h]
  _BYTE *v104; // [rsp+70h] [rbp-240h]
  _BYTE *v105; // [rsp+70h] [rbp-240h]
  unsigned int *v106; // [rsp+70h] [rbp-240h]
  unsigned int *v107; // [rsp+78h] [rbp-238h]
  _BYTE s[560]; // [rsp+80h] [rbp-230h] BYREF

  v6 = 2 * a2;
  v7 = 2 * a4;
  LODWORD(v8) = 2 * a2 - v7;
  v9 = 2 * v8;
  v10 = (unsigned int)(v6 + 1);
  v11 = 4 * v10;
  if ( a6 )
  {
    if ( v9 + 8 * a4 + 1 > 0x80 )
    {
      v98 = 4 * v10;
      v100 = sub_2207820(4 * v10);
      v107 = (unsigned int *)sub_2207820(4 * v7);
      v96 = (_BYTE *)sub_2207820(4 * v6);
      v83 = sub_2207820(4 * v7);
      v12 = (_BYTE *)v100;
      v11 = v98;
      v91 = (_DWORD *)v83;
    }
    else
    {
      v12 = s;
      v107 = (unsigned int *)&s[v11];
      v96 = &s[4 * (unsigned int)(v7 + v6 + 1)];
      v91 = &s[4 * (unsigned int)(v10 + v7 + v6)];
    }
  }
  else if ( v9 + 2 * ((unsigned int)v7 + a4) + 1 > 0x80 )
  {
    v99 = 4 * v10;
    v101 = sub_2207820(4 * v10);
    v107 = (unsigned int *)sub_2207820(4 * v7);
    v84 = sub_2207820(4 * v6);
    v12 = (_BYTE *)v101;
    v91 = 0;
    v96 = (_BYTE *)v84;
    v11 = v99;
  }
  else
  {
    v91 = 0;
    v12 = s;
    v107 = (unsigned int *)&s[v11];
    v96 = &s[4 * (unsigned int)(v7 + v10)];
  }
  v13 = memset(v12, 0, v11);
  if ( a2 )
  {
    v14 = 0;
    v15 = (__int64)&a1[a2];
    do
    {
      v16 = *a1++;
      v13[v14] = v16;
      v17 = (unsigned int)(v14 + 1);
      v14 += 2;
      v13[v17] = HIDWORD(v16);
    }
    while ( (__int64 *)v15 != a1 );
  }
  v13[v6] = 0;
  v104 = v13;
  memset(v107, 0, 4 * v7);
  if ( !a4 )
  {
    memset(v96, 0, 4 * v6);
    v22 = v104;
    if ( !a6 )
    {
LABEL_99:
      if ( (_DWORD)v8 )
      {
        v95 = 0;
        v26 = v8;
        goto LABEL_18;
      }
      goto LABEL_103;
    }
    goto LABEL_11;
  }
  v18 = a3;
  v19 = 0;
  do
  {
    v20 = *v18++;
    v107[v19] = v20;
    v21 = (unsigned int)(v19 + 1);
    v19 += 2;
    v107[v21] = HIDWORD(v20);
  }
  while ( &a3[a4] != v18 );
  memset(v96, 0, 4 * v6);
  v22 = v104;
  if ( a6 )
  {
LABEL_11:
    v105 = v22;
    memset(v91, 0, 4 * v7);
    v22 = v105;
  }
  if ( !(_DWORD)v7 )
    goto LABEL_99;
  v23 = (unsigned int)(v7 - 1);
  v24 = &v107[v23];
  while ( 1 )
  {
    v25 = v7;
    LODWORD(v7) = v23;
    if ( *v24 )
    {
      v95 = v25;
      v26 = v8 + v25;
      if ( (_DWORD)v8 + v25 )
      {
LABEL_18:
        for ( i = &v22[4 * (v26 - 1)]; !*i; --i )
        {
          LODWORD(v8) = v8 - 1;
          if ( v22 == (_BYTE *)i )
            break;
        }
      }
      v28 = v8;
      if ( v95 != 1 )
      {
        v29 = (unsigned int)(v95 + v8);
        v30 = &v22[4 * v29];
        v85 = v95 - 1;
        v31 = (unsigned int)(v95 - 2);
        v32 = v107[v95 - 1];
        v93 = &v107[v95 - 1];
        if ( v32 )
          goto LABEL_24;
        v89 = 32;
        v86 = 32;
        goto LABEL_87;
      }
      v69 = *v107;
      if ( (int)v8 < 0 )
      {
        LODWORD(v70) = 0;
LABEL_72:
        result = (unsigned __int64)v91;
        if ( v91 )
          *v91 = v70;
        goto LABEL_74;
      }
      v8 = (int)v8;
      v70 = 0;
      while ( 1 )
      {
        while ( 1 )
        {
          v70 = *(unsigned int *)&v22[4 * v8] | (v70 << 32);
          if ( v70 )
            break;
          *(_DWORD *)&v96[4 * v8] = 0;
          v70 = 0;
LABEL_68:
          if ( (int)--v8 < 0 )
            goto LABEL_72;
        }
        if ( v69 <= v70 )
        {
          if ( v69 == v70 )
          {
            *(_DWORD *)&v96[4 * v8] = 1;
            v70 = 0;
          }
          else
          {
            *(_DWORD *)&v96[4 * v8] = v70 / v69;
            v70 %= v69;
          }
          goto LABEL_68;
        }
        *(_DWORD *)&v96[4 * v8--] = 0;
        if ( (int)v8 < 0 )
          goto LABEL_72;
      }
    }
    LODWORD(v8) = v8 + 1;
    --v24;
    if ( !(_DWORD)v23 )
      break;
    LODWORD(v23) = v23 - 1;
  }
  if ( (_DWORD)v8 )
  {
    v95 = 0;
    v26 = v8;
    goto LABEL_18;
  }
LABEL_103:
  v30 = v22;
  v32 = v107[0xFFFFFFFFLL];
  v93 = v107 + 0xFFFFFFFFLL;
  v29 = 0;
  if ( !v32 )
  {
    v85 = -1;
    v28 = 0;
    v31 = 4294967294LL;
    v86 = 32;
    v89 = 32;
LABEL_106:
    v95 = 0;
    goto LABEL_26;
  }
  v85 = -1;
  v28 = 0;
  v31 = 4294967294LL;
  v95 = 0;
LABEL_24:
  _BitScanReverse(&v32, v32);
  v89 = v32 ^ 0x1F;
  v86 = (int)(v32 ^ 0x1F);
  if ( v32 == 0x1F )
  {
    v86 = 0;
    v29 = 0;
    goto LABEL_26;
  }
LABEL_87:
  if ( (_DWORD)v29 )
  {
    v75 = (unsigned int *)v22;
    v76 = (__int64)&v22[4 * (unsigned int)(v29 - 1) + 4];
    LODWORD(v29) = 0;
    do
    {
      v77 = *v75;
      v78 = v29;
      ++v75;
      v29 = v77 >> (32 - v89);
      *(v75 - 1) = v78 | (v77 << v89);
    }
    while ( (unsigned int *)v76 != v75 );
  }
  if ( !v95 )
    goto LABEL_106;
  v79 = v107;
  v80 = 0;
  do
  {
    v81 = *v79;
    v82 = v80;
    ++v79;
    v80 = v81 >> (32 - v89);
    *(v79 - 1) = v82 | (v81 << v89);
  }
  while ( &v107[v95 - 1 + 1] != v79 );
LABEL_26:
  *v30 = v29;
  v33 = v28;
  v34 = v95 + v28;
  v92 = &v107[v31];
  v103 = v95 + v28 - 2;
  while ( 2 )
  {
    v35 = *v93;
    v97 = (unsigned int *)&v22[4 * (v34 - 1)];
    v106 = (unsigned int *)&v22[4 * v34];
    v38 = *v97 | (unsigned __int64)(v29 << 32);
    v36 = v38 / v35;
    v37 = v38 % v35;
    v39 = v38 / v35;
    if ( v38 / v35 == 0x100000000LL )
    {
      v43 = v35 + v37;
      v39 = 0xFFFFFFFFLL;
      if ( v43 <= 0xFFFFFFFF )
      {
        v42 = 0xFFFFFFFFLL;
        v40 = *v92;
        v41 = *(unsigned int *)&v22[4 * v103];
        goto LABEL_41;
      }
      goto LABEL_32;
    }
    v40 = *v92;
    v41 = *(unsigned int *)&v22[4 * v103];
    if ( v40 * v36 <= v41 + (v37 << 32) || (v42 = v36 - 1, v43 = v35 + v37, v39 = v36 - 1, v43 > 0xFFFFFFFF) )
    {
LABEL_32:
      v44 = v39;
      if ( v95 )
        goto LABEL_33;
LABEL_43:
      result = (unsigned __int64)v96;
      --v103;
      *(_DWORD *)&v96[4 * v33--] = v39;
      if ( (int)v33 < 0 )
        goto LABEL_44;
LABEL_38:
      --v34;
      v29 = *v97;
      continue;
    }
    break;
  }
  if ( v36 == 0x100000001LL )
    goto LABEL_31;
LABEL_41:
  v39 = v42;
  if ( v42 * v40 > (v43 << 32) + v41 )
  {
LABEL_31:
    v39 = v36 - 2;
    goto LABEL_32;
  }
  v44 = v42;
  if ( !v95 )
    goto LABEL_43;
LABEL_33:
  v45 = v107;
  v46 = v33;
  v47 = v33;
  v48 = 0;
  v49 = v107;
  do
  {
    v50 = *v45;
    v51 = v47++;
    ++v45;
    v52 = &v22[4 * v51];
    v54 = v39 * v50;
    v53 = (unsigned int)*v52 - v48 - (unsigned int)v54;
    *v52 = v53;
    v53 >>= 32;
    LODWORD(v54) = HIDWORD(v54) - v53;
    v55 = v53;
    v48 = (unsigned int)(HIDWORD(v54) - v53);
  }
  while ( v34 != v47 );
  v56 = *v106;
  *v106 = *v106 + v55 - HIDWORD(v54);
  if ( v56 < (unsigned int)v54 )
  {
    v65 = 0;
    *(_DWORD *)&v96[4 * v33] = v44 - 1;
    do
    {
      while ( 1 )
      {
        v66 = (unsigned int *)&v22[4 * v46];
        v67 = *v49;
        if ( *v66 <= *v49 )
          v67 = *v66;
        v68 = *v66 + *v49 + v65;
        *v66 = v68;
        if ( v68 < v67 )
          break;
        ++v46;
        ++v49;
        v65 &= v68 == v67;
        result = v65;
        if ( v34 == v46 )
          goto LABEL_64;
      }
      ++v46;
      result = 1;
      v65 = 1;
      ++v49;
    }
    while ( v34 != v46 );
LABEL_64:
    *v106 += result;
  }
  else
  {
    result = (unsigned __int64)v96;
    *(_DWORD *)&v96[4 * v33] = v44;
  }
  --v33;
  --v103;
  if ( (int)v33 >= 0 )
    goto LABEL_38;
LABEL_44:
  if ( v91 )
  {
    if ( v86 )
    {
      result = (unsigned int)v85;
      if ( v85 >= 0 )
      {
        v58 = v85;
        v59 = 0;
        do
        {
          result = v59 | (*(_DWORD *)&v22[4 * v58] >> v89);
          v91[v58] = result;
          v60 = *(_DWORD *)&v22[4 * v58--];
          v59 = v60 << (32 - v89);
        }
        while ( (int)v58 >= 0 );
        if ( !a5 )
          goto LABEL_50;
        goto LABEL_75;
      }
    }
    else
    {
      result = v85;
      if ( v85 >= 0 )
      {
        do
        {
          v91[result] = *(_DWORD *)&v22[4 * result];
          --result;
        }
        while ( (result & 0x80000000) == 0LL );
      }
    }
  }
LABEL_74:
  if ( !a5 )
    goto LABEL_50;
LABEL_75:
  result = a2;
  if ( a2 )
  {
    v71 = a5;
    v72 = 0;
    do
    {
      v73 = v72 + 1;
      v74 = v72;
      v71 += 8;
      v72 += 2;
      result = *(unsigned int *)&v96[4 * v74] | ((unsigned __int64)*(unsigned int *)&v96[4 * v73] << 32);
      *(_QWORD *)(v71 - 8) = result;
    }
    while ( a5 + 8LL * (a2 - 1) + 8 != v71 );
  }
LABEL_50:
  if ( a6 )
  {
    result = a4;
    if ( a4 )
    {
      v61 = a6;
      v62 = 0;
      do
      {
        v63 = v62 + 1;
        v64 = v62;
        v61 += 8;
        v62 += 2;
        result = (unsigned int)v91[v64] | ((unsigned __int64)(unsigned int)v91[v63] << 32);
        *(_QWORD *)(v61 - 8) = result;
      }
      while ( a6 + 8LL * (a4 - 1) + 8 != v61 );
    }
  }
  if ( v22 != s )
  {
    j_j___libc_free_0_0(v22);
    j_j___libc_free_0_0(v107);
    j_j___libc_free_0_0(v96);
    result = (unsigned __int64)v91;
    if ( v91 )
      return j_j___libc_free_0_0(v91);
  }
  return result;
}
