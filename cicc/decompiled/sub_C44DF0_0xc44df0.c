// Function: sub_C44DF0
// Address: 0xc44df0
//
unsigned __int64 __fastcall sub_C44DF0(
        __int64 *a1,
        unsigned int a2,
        __int64 *a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6)
{
  __int64 v6; // r15
  __int64 v7; // r14
  int v8; // ebx
  int v9; // esi
  __int64 v10; // rcx
  size_t v11; // rdx
  _BYTE *v12; // r10
  _DWORD *v13; // r10
  int v14; // edx
  __int64 v15; // rsi
  __int64 v16; // rax
  __int64 v17; // rcx
  __int64 *v18; // rdx
  int v19; // ecx
  __int64 v20; // rax
  __int64 v21; // r8
  _BYTE *v22; // r10
  __int64 v23; // rax
  unsigned int *v24; // rdx
  int v25; // ecx
  int v26; // eax
  _DWORD *i; // rax
  unsigned int v28; // edi
  __int64 v29; // r11
  unsigned __int64 v30; // rcx
  unsigned __int64 result; // rax
  __int64 v32; // r8
  unsigned int v33; // edx
  __int64 v34; // rax
  __int64 v35; // rdi
  __int64 v36; // r9
  unsigned int v37; // edx
  __int64 v38; // rax
  __int64 v39; // rdi
  __int64 v40; // rax
  _DWORD *v41; // r12
  __int64 v42; // r13
  unsigned int v43; // edx
  int v44; // edx
  __int64 v45; // r11
  int v46; // r9d
  unsigned __int64 v47; // rcx
  unsigned __int64 v48; // rax
  unsigned __int64 v49; // rdx
  unsigned __int64 v50; // rtt
  __int64 v51; // rdi
  __int64 v52; // r8
  __int64 v53; // rbx
  __int64 v54; // rsi
  unsigned __int64 v55; // rdx
  int v56; // r12d
  unsigned int *v57; // rsi
  int v58; // r8d
  unsigned int v59; // ecx
  __int64 v60; // r15
  unsigned int *v61; // rbx
  __int64 v62; // rax
  __int64 v63; // rdx
  _DWORD *v64; // r13
  unsigned __int64 v65; // rdx
  __int64 v66; // rax
  int v67; // r14d
  unsigned int v68; // edx
  __int64 v69; // rax
  unsigned int *v70; // rsi
  __int64 v71; // r9
  unsigned int v72; // edx
  int v73; // edi
  unsigned int *v74; // rsi
  unsigned int v75; // edi
  unsigned int v76; // edx
  unsigned int v77; // r14d
  __int64 v78; // rsi
  unsigned int v79; // edi
  int v80; // edi
  unsigned __int8 v81; // si
  unsigned int *v82; // rdi
  unsigned int v83; // r12d
  unsigned int v84; // eax
  __int64 v85; // rax
  int v86; // [rsp+Ch] [rbp-2A4h]
  int v89; // [rsp+28h] [rbp-288h]
  _DWORD *v91; // [rsp+30h] [rbp-280h]
  unsigned int *v92; // [rsp+38h] [rbp-278h]
  unsigned int *v93; // [rsp+40h] [rbp-270h]
  _BYTE *v94; // [rsp+48h] [rbp-268h]
  int v96; // [rsp+54h] [rbp-25Ch]
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
  v8 = 2 * a2 - v7;
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
      v94 = (_BYTE *)sub_2207820(4 * v6);
      v69 = sub_2207820(4 * v7);
      v12 = (_BYTE *)v100;
      v11 = v98;
      v91 = (_DWORD *)v69;
    }
    else
    {
      v12 = s;
      v107 = (unsigned int *)&s[v11];
      v94 = &s[4 * (unsigned int)(v7 + v6 + 1)];
      v91 = &s[4 * (unsigned int)(v10 + v7 + v6)];
    }
  }
  else if ( v9 + 2 * ((unsigned int)v7 + a4) + 1 > 0x80 )
  {
    v99 = 4 * v10;
    v101 = sub_2207820(4 * v10);
    v107 = (unsigned int *)sub_2207820(4 * v7);
    v85 = sub_2207820(4 * v6);
    v12 = (_BYTE *)v101;
    v91 = 0;
    v94 = (_BYTE *)v85;
    v11 = v99;
  }
  else
  {
    v91 = 0;
    v12 = s;
    v107 = (unsigned int *)&s[v11];
    v94 = &s[4 * (unsigned int)(v7 + v10)];
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
    memset(v94, 0, 4 * v6);
    v22 = v104;
    if ( !a6 )
      goto LABEL_98;
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
  memset(v94, 0, 4 * v6);
  v22 = v104;
  if ( a6 )
  {
LABEL_11:
    v105 = v22;
    memset(v91, 0, 4 * v7);
    v22 = v105;
  }
  if ( !(_DWORD)v7 )
  {
LABEL_98:
    v96 = 0;
    v26 = v8;
    if ( v8 )
      goto LABEL_18;
    v41 = v22;
    v42 = 4294967294LL;
    v43 = v107[0xFFFFFFFFLL];
    v93 = v107 + 0xFFFFFFFFLL;
    v40 = 0;
    if ( !v43 )
    {
      v86 = -1;
      v8 = 0;
      v89 = 32;
      goto LABEL_49;
    }
LABEL_100:
    v86 = -1;
    goto LABEL_47;
  }
  v23 = (unsigned int)(v7 - 1);
  v24 = &v107[v23];
  while ( 1 )
  {
    v25 = v7;
    LODWORD(v7) = v23;
    if ( *v24 )
      break;
    ++v8;
    --v24;
    if ( !(_DWORD)v23 )
    {
      if ( v8 )
      {
        v96 = 0;
        v26 = v8;
        goto LABEL_18;
      }
      v41 = v22;
      v43 = v107[0xFFFFFFFFLL];
      v93 = v107 + 0xFFFFFFFFLL;
      v40 = 0;
      if ( v43 )
      {
        v96 = 0;
        v8 = 0;
        v42 = 4294967294LL;
        goto LABEL_100;
      }
      v86 = -1;
      v8 = 0;
      v42 = 4294967294LL;
      v89 = 32;
LABEL_105:
      v96 = 0;
LABEL_49:
      v45 = v8;
      *v41 = v40;
      v92 = &v107[v42];
      v46 = v96 + v8;
      v103 = v96 + v8 - 2;
      while ( 2 )
      {
        v47 = *v93;
        v97 = (unsigned int *)&v22[4 * (v46 - 1)];
        v106 = (unsigned int *)&v22[4 * v46];
        v50 = *v97 | (unsigned __int64)(v40 << 32);
        v48 = v50 / v47;
        v49 = v50 % v47;
        v51 = v50 / v47;
        if ( v50 / v47 == 0x100000000LL )
        {
          v55 = v47 + v49;
          v51 = 0xFFFFFFFFLL;
          if ( v55 <= 0xFFFFFFFF )
          {
            v54 = 0xFFFFFFFFLL;
            v52 = *v92;
            v53 = *(unsigned int *)&v22[4 * v103];
            goto LABEL_78;
          }
        }
        else
        {
          v52 = *v92;
          v53 = *(unsigned int *)&v22[4 * v103];
          if ( v52 * v48 > v53 + (v49 << 32) )
          {
            v54 = v48 - 1;
            v55 = v47 + v49;
            v51 = v48 - 1;
            if ( v55 <= 0xFFFFFFFF )
            {
              if ( v48 == 0x100000001LL )
                goto LABEL_54;
LABEL_78:
              v51 = v54;
              if ( v54 * v52 <= (v55 << 32) + v53 )
              {
                v56 = v54;
                if ( !v96 )
                  goto LABEL_80;
LABEL_56:
                v57 = v107;
                v58 = v45;
                v59 = v45;
                v60 = 0;
                v61 = v107;
                do
                {
                  v62 = *v57;
                  v63 = v59++;
                  ++v57;
                  v64 = &v22[4 * v63];
                  v66 = v51 * v62;
                  v65 = (unsigned int)*v64 - v60 - (unsigned int)v66;
                  *v64 = v65;
                  v65 >>= 32;
                  LODWORD(v66) = HIDWORD(v66) - v65;
                  v67 = v65;
                  v60 = (unsigned int)(HIDWORD(v66) - v65);
                }
                while ( v46 != v59 );
                v68 = *v106;
                *v106 = *v106 + v67 - HIDWORD(v66);
                if ( v68 < (unsigned int)v66 )
                {
                  v81 = 0;
                  *(_DWORD *)&v94[4 * v45] = v56 - 1;
                  do
                  {
                    while ( 1 )
                    {
                      v82 = (unsigned int *)&v22[4 * v58];
                      v83 = *v61;
                      if ( *v82 <= *v61 )
                        v83 = *v82;
                      v84 = *v82 + *v61 + v81;
                      *v82 = v84;
                      if ( v83 > v84 )
                        break;
                      ++v58;
                      ++v61;
                      v81 &= v83 == v84;
                      result = v81;
                      if ( v46 == v58 )
                        goto LABEL_93;
                    }
                    ++v58;
                    result = 1;
                    v81 = 1;
                    ++v61;
                  }
                  while ( v46 != v58 );
LABEL_93:
                  *v106 += result;
                }
                else
                {
                  result = (unsigned __int64)v94;
                  *(_DWORD *)&v94[4 * v45] = v56;
                }
                --v45;
                --v103;
                if ( (int)v45 < 0 )
                {
LABEL_81:
                  if ( v91 )
                  {
                    if ( v89 )
                    {
                      result = (unsigned int)v86;
                      if ( v86 >= 0 )
                      {
                        v78 = v86;
                        v79 = 0;
                        do
                        {
                          result = v79 | (*(_DWORD *)&v22[4 * v78] >> v89);
                          v91[v78] = result;
                          v80 = *(_DWORD *)&v22[4 * v78--];
                          v79 = v80 << (32 - v89);
                        }
                        while ( (int)v78 >= 0 );
                      }
                    }
                    else
                    {
                      result = v86;
                      if ( v86 >= 0 )
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
                  goto LABEL_32;
                }
                goto LABEL_61;
              }
LABEL_54:
              v51 = v48 - 2;
            }
          }
        }
        v56 = v51;
        if ( v96 )
          goto LABEL_56;
LABEL_80:
        result = (unsigned __int64)v94;
        --v103;
        *(_DWORD *)&v94[4 * v45--] = v51;
        if ( (int)v45 < 0 )
          goto LABEL_81;
LABEL_61:
        --v46;
        v40 = *v97;
        continue;
      }
    }
    LODWORD(v23) = v23 - 1;
  }
  v96 = v25;
  v26 = v8 + v25;
  if ( v8 + v25 )
  {
LABEL_18:
    for ( i = &v22[4 * (v26 - 1)]; !*i; --i )
    {
      --v8;
      if ( v22 == (_BYTE *)i )
        break;
    }
  }
  if ( v96 != 1 )
  {
    v40 = (unsigned int)(v96 + v8);
    v41 = &v22[4 * v40];
    v86 = v96 - 1;
    v42 = (unsigned int)(v96 - 2);
    v43 = v107[v96 - 1];
    v93 = &v107[v96 - 1];
    if ( !v43 )
    {
      v89 = 32;
LABEL_69:
      if ( (_DWORD)v40 )
      {
        v70 = (unsigned int *)v22;
        v71 = (__int64)&v22[4 * (unsigned int)(v40 - 1) + 4];
        LODWORD(v40) = 0;
        do
        {
          v72 = *v70;
          v73 = v40;
          ++v70;
          v40 = v72 >> (32 - v89);
          *(v70 - 1) = v73 | (v72 << v89);
        }
        while ( (unsigned int *)v71 != v70 );
      }
      if ( !v96 )
        goto LABEL_105;
      v74 = v107;
      v75 = 0;
      do
      {
        v76 = *v74;
        v77 = v75;
        ++v74;
        v75 = v76 >> (32 - v89);
        *(v74 - 1) = v77 | (v76 << v89);
      }
      while ( &v107[v96 - 1 + 1] != v74 );
      goto LABEL_49;
    }
LABEL_47:
    _BitScanReverse(&v43, v43);
    v44 = v43 ^ 0x1F;
    if ( !v44 )
    {
      v89 = 0;
      v40 = 0;
      goto LABEL_49;
    }
    v89 = v44;
    goto LABEL_69;
  }
  v28 = *v107;
  if ( v8 >= 0 )
  {
    v29 = v8;
    v30 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v30 = *(unsigned int *)&v22[4 * v29] | (v30 << 32);
        if ( v30 )
          break;
        *(_DWORD *)&v94[4 * v29] = 0;
        v30 = 0;
LABEL_26:
        if ( (int)--v29 < 0 )
          goto LABEL_30;
      }
      if ( v28 <= v30 )
      {
        if ( v28 == v30 )
        {
          *(_DWORD *)&v94[4 * v29] = 1;
          v30 = 0;
        }
        else
        {
          *(_DWORD *)&v94[4 * v29] = v30 / v28;
          v30 %= v28;
        }
        goto LABEL_26;
      }
      *(_DWORD *)&v94[4 * v29--] = 0;
      if ( (int)v29 < 0 )
        goto LABEL_30;
    }
  }
  LODWORD(v30) = 0;
LABEL_30:
  result = (unsigned __int64)v91;
  if ( v91 )
    *v91 = v30;
LABEL_32:
  if ( a5 )
  {
    result = a2;
    if ( a2 )
    {
      v32 = a5;
      v33 = 0;
      do
      {
        v34 = v33 + 1;
        v35 = v33;
        v32 += 8;
        v33 += 2;
        result = *(unsigned int *)&v94[4 * v35] | ((unsigned __int64)*(unsigned int *)&v94[4 * v34] << 32);
        *(_QWORD *)(v32 - 8) = result;
      }
      while ( a5 + 8LL * (a2 - 1) + 8 != v32 );
    }
  }
  if ( a6 )
  {
    result = a4;
    if ( a4 )
    {
      v36 = a6;
      v37 = 0;
      do
      {
        v38 = v37 + 1;
        v39 = v37;
        v36 += 8;
        v37 += 2;
        result = (unsigned int)v91[v39] | ((unsigned __int64)(unsigned int)v91[v38] << 32);
        *(_QWORD *)(v36 - 8) = result;
      }
      while ( v36 != a6 + 8LL * (a4 - 1) + 8 );
    }
  }
  if ( v22 != s )
  {
    j_j___libc_free_0_0(v22);
    j_j___libc_free_0_0(v107);
    j_j___libc_free_0_0(v94);
    result = (unsigned __int64)v91;
    if ( v91 )
      return j_j___libc_free_0_0(v91);
  }
  return result;
}
