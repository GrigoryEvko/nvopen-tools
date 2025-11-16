// Function: sub_1630FC0
// Address: 0x1630fc0
//
__int64 __fastcall sub_1630FC0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 *v4; // rcx
  unsigned __int64 *v5; // rax
  __int64 v6; // rax
  _BYTE *v7; // r8
  _BYTE *v8; // r9
  __int64 *v9; // r12
  __int64 v10; // rsi
  _QWORD *v11; // rdi
  _QWORD *v12; // rax
  _QWORD *v13; // rcx
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 v16; // r13
  __int64 *v17; // r15
  __int64 v18; // r13
  __int64 *v19; // r14
  __int64 v20; // r13
  __int64 v21; // r13
  __int64 v22; // r14
  _QWORD *v23; // rdi
  int v24; // ecx
  unsigned int v25; // eax
  __int64 *v26; // rsi
  __int64 v27; // r8
  __int64 *v28; // r13
  __int64 *v29; // rbx
  __int64 v30; // r14
  _QWORD *v31; // r8
  int v32; // ecx
  unsigned int v33; // eax
  __int64 *v34; // rsi
  __int64 v35; // rdi
  __int64 *v36; // rax
  __int64 *v37; // rdi
  __int64 v38; // r12
  _QWORD *v40; // r8
  int v41; // edi
  unsigned int v42; // eax
  __int64 *v43; // rdx
  __int64 v44; // rsi
  _QWORD *v45; // rdi
  int v46; // esi
  unsigned int v47; // eax
  __int64 v48; // r8
  int v49; // edx
  _QWORD *v50; // rdi
  int v51; // esi
  unsigned int v52; // eax
  __int64 v53; // r8
  int v54; // edx
  int v55; // ecx
  __int64 v56; // r13
  _QWORD *v57; // rdi
  int v58; // ecx
  unsigned int v59; // eax
  __int64 v60; // r9
  int v61; // esi
  int v62; // r8d
  int v63; // esi
  int v64; // r9d
  int v65; // edx
  int v66; // ecx
  int v67; // esi
  int v68; // r9d
  __int64 v69; // r13
  __int64 v70; // r13
  _QWORD *v71; // rdi
  int v72; // ecx
  unsigned int v73; // eax
  __int64 v74; // r9
  int v75; // esi
  int v76; // r8d
  int v77; // ecx
  _QWORD *v78; // rdi
  int v79; // ecx
  unsigned int v80; // eax
  __int64 v81; // r9
  int v82; // esi
  int v83; // r8d
  __int64 *v85; // [rsp+18h] [rbp-118h]
  __int64 *v86; // [rsp+20h] [rbp-110h]
  __int64 *v87; // [rsp+30h] [rbp-100h]
  __int64 *v88; // [rsp+38h] [rbp-F8h]
  __int64 v89; // [rsp+48h] [rbp-E8h] BYREF
  __int64 v90; // [rsp+50h] [rbp-E0h] BYREF
  _BYTE *v91; // [rsp+58h] [rbp-D8h]
  _BYTE *v92; // [rsp+60h] [rbp-D0h]
  __int64 v93; // [rsp+68h] [rbp-C8h]
  int i; // [rsp+70h] [rbp-C0h]
  _BYTE v95[40]; // [rsp+78h] [rbp-B8h] BYREF
  __int64 v96; // [rsp+A0h] [rbp-90h] BYREF
  __int64 v97; // [rsp+A8h] [rbp-88h]
  _QWORD *v98; // [rsp+B0h] [rbp-80h] BYREF
  int v99; // [rsp+B8h] [rbp-78h]
  __int64 *v100; // [rsp+D0h] [rbp-60h] BYREF
  __int64 v101; // [rsp+D8h] [rbp-58h]
  _BYTE v102[80]; // [rsp+E0h] [rbp-50h] BYREF

  if ( !a1 || !a2 )
    return 0;
  v3 = *(unsigned int *)(a1 + 8);
  v96 = 0;
  v97 = 1;
  v4 = (__int64 *)(a1 - 8 * v3);
  v5 = (unsigned __int64 *)&v98;
  do
    *v5++ = -4;
  while ( v5 != (unsigned __int64 *)&v100 );
  v100 = (__int64 *)v102;
  v101 = 0x400000000LL;
  sub_1630C80((__int64)&v96, v4, (__int64 *)a1);
  v6 = *(unsigned int *)(a2 + 8);
  v7 = v95;
  v90 = 0;
  v91 = v95;
  v8 = v95;
  v92 = v95;
  v93 = 4;
  v9 = (__int64 *)(a2 - 8 * v6);
  for ( i = 0; (__int64 *)a2 != v9; ++v9 )
  {
LABEL_9:
    v10 = *v9;
    if ( v7 != v8 )
      goto LABEL_7;
    v11 = &v7[8 * HIDWORD(v93)];
    if ( v11 != (_QWORD *)v7 )
    {
      v12 = v7;
      v13 = 0;
      while ( v10 != *v12 )
      {
        if ( *v12 == -2 )
          v13 = v12;
        if ( v11 == ++v12 )
        {
          if ( !v13 )
            goto LABEL_78;
          ++v9;
          *v13 = v10;
          v8 = v92;
          --i;
          v7 = v91;
          ++v90;
          if ( (__int64 *)a2 != v9 )
            goto LABEL_9;
          goto LABEL_18;
        }
      }
      continue;
    }
LABEL_78:
    if ( HIDWORD(v93) < (unsigned int)v93 )
    {
      ++HIDWORD(v93);
      *v11 = v10;
      v7 = v91;
      ++v90;
      v8 = v92;
    }
    else
    {
LABEL_7:
      sub_16CCBA0(&v90, v10);
      v8 = v92;
      v7 = v91;
    }
  }
LABEL_18:
  v14 = 8LL * (unsigned int)v101;
  v86 = v100;
  v85 = (__int64 *)(unsigned int)v101;
  v87 = &v100[(unsigned __int64)v14 / 8];
  v15 = v14 >> 3;
  v16 = v14 >> 5;
  if ( !v16 )
  {
    v17 = v100;
LABEL_82:
    if ( v15 != 2 )
    {
      if ( v15 != 3 )
      {
        if ( v15 != 1 )
          goto LABEL_42;
LABEL_85:
        v56 = *v17;
        v89 = *v17;
        if ( sub_161FC80((__int64)&v90, &v89) )
          goto LABEL_42;
        if ( (v97 & 1) != 0 )
        {
          v57 = &v98;
          v58 = 3;
        }
        else
        {
          v57 = v98;
          if ( !v99 )
            goto LABEL_29;
          v58 = v99 - 1;
        }
        v59 = v58 & (((unsigned int)v56 >> 9) ^ ((unsigned int)v56 >> 4));
        v26 = &v57[v59];
        v60 = *v26;
        if ( v56 != *v26 )
        {
          v61 = 1;
          while ( v60 != -4 )
          {
            v62 = v61 + 1;
            v59 = v58 & (v61 + v59);
            v26 = &v57[v59];
            v60 = *v26;
            if ( v56 == *v26 )
              goto LABEL_28;
            v61 = v62;
          }
          goto LABEL_29;
        }
        goto LABEL_28;
      }
      v69 = *v17;
      v89 = *v17;
      if ( !sub_161FC80((__int64)&v90, &v89) )
      {
        if ( (v97 & 1) != 0 )
        {
          v78 = &v98;
          v79 = 3;
        }
        else
        {
          v78 = v98;
          if ( !v99 )
            goto LABEL_29;
          v79 = v99 - 1;
        }
        v80 = v79 & (((unsigned int)v69 >> 9) ^ ((unsigned int)v69 >> 4));
        v26 = &v78[v80];
        v81 = *v26;
        if ( v69 != *v26 )
        {
          v82 = 1;
          while ( v81 != -4 )
          {
            v83 = v82 + 1;
            v80 = v79 & (v82 + v80);
            v26 = &v78[v80];
            v81 = *v26;
            if ( v69 == *v26 )
              goto LABEL_28;
            v82 = v83;
          }
          goto LABEL_29;
        }
        goto LABEL_28;
      }
      ++v17;
    }
    v70 = *v17;
    v89 = *v17;
    if ( sub_161FC80((__int64)&v90, &v89) )
    {
      ++v17;
      goto LABEL_85;
    }
    if ( (v97 & 1) != 0 )
    {
      v71 = &v98;
      v72 = 3;
    }
    else
    {
      v71 = v98;
      if ( !v99 )
        goto LABEL_29;
      v72 = v99 - 1;
    }
    v73 = v72 & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
    v26 = &v71[v73];
    v74 = *v26;
    if ( v70 != *v26 )
    {
      v75 = 1;
      while ( v74 != -4 )
      {
        v76 = v75 + 1;
        v73 = v72 & (v75 + v73);
        v26 = &v71[v73];
        v74 = *v26;
        if ( v70 == *v26 )
          goto LABEL_28;
        v75 = v76;
      }
      goto LABEL_29;
    }
LABEL_28:
    *v26 = -8;
    ++HIDWORD(v97);
    LODWORD(v97) = (2 * ((unsigned int)v97 >> 1) - 2) | v97 & 1;
    goto LABEL_29;
  }
  v17 = v100;
  v88 = &v100[4 * v16];
  while ( 1 )
  {
    v22 = *v17;
    v89 = *v17;
    if ( !sub_161FC80((__int64)&v90, &v89) )
      break;
    v18 = v17[1];
    v19 = v17 + 1;
    v89 = v18;
    if ( !sub_161FC80((__int64)&v90, &v89) )
    {
      if ( (v97 & 1) != 0 )
      {
        v40 = &v98;
        v41 = 3;
      }
      else
      {
        v40 = v98;
        ++v17;
        if ( !v99 )
          goto LABEL_29;
        v41 = v99 - 1;
      }
      v42 = v41 & (((unsigned int)v18 >> 9) ^ ((unsigned int)v18 >> 4));
      v43 = &v40[v42];
      v44 = *v43;
      if ( v18 == *v43 )
        goto LABEL_56;
      v65 = 1;
      while ( v44 != -4 )
      {
        v66 = v65 + 1;
        v42 = v41 & (v65 + v42);
        v43 = &v40[v42];
        v44 = *v43;
        if ( v18 == *v43 )
          goto LABEL_56;
        v65 = v66;
      }
LABEL_62:
      v17 = v19;
      goto LABEL_29;
    }
    v20 = v17[2];
    v19 = v17 + 2;
    v89 = v20;
    if ( !sub_161FC80((__int64)&v90, &v89) )
    {
      if ( (v97 & 1) != 0 )
      {
        v45 = &v98;
        v46 = 3;
      }
      else
      {
        v45 = v98;
        v17 += 2;
        if ( !v99 )
          goto LABEL_29;
        v46 = v99 - 1;
      }
      v47 = v46 & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
      v43 = &v45[v47];
      v48 = *v43;
      if ( v20 == *v43 )
        goto LABEL_56;
      v49 = 1;
      while ( v48 != -4 )
      {
        v77 = v49 + 1;
        v47 = v46 & (v49 + v47);
        v43 = &v45[v47];
        v48 = *v43;
        if ( v20 == *v43 )
          goto LABEL_56;
        v49 = v77;
      }
      goto LABEL_62;
    }
    v21 = v17[3];
    v19 = v17 + 3;
    v89 = v21;
    if ( !sub_161FC80((__int64)&v90, &v89) )
    {
      if ( (v97 & 1) != 0 )
      {
        v50 = &v98;
        v51 = 3;
        goto LABEL_65;
      }
      v50 = v98;
      v17 += 3;
      if ( v99 )
      {
        v51 = v99 - 1;
LABEL_65:
        v52 = v51 & (((unsigned int)v21 >> 9) ^ ((unsigned int)v21 >> 4));
        v43 = &v50[v52];
        v53 = *v43;
        if ( v21 != *v43 )
        {
          v54 = 1;
          while ( v53 != -4 )
          {
            v55 = v54 + 1;
            v52 = v51 & (v54 + v52);
            v43 = &v50[v52];
            v53 = *v43;
            if ( v21 == *v43 )
              goto LABEL_56;
            v54 = v55;
          }
          goto LABEL_62;
        }
LABEL_56:
        *v43 = -8;
        v17 = v19;
        ++HIDWORD(v97);
        LODWORD(v97) = (2 * ((unsigned int)v97 >> 1) - 2) | v97 & 1;
        goto LABEL_29;
      }
      goto LABEL_29;
    }
    v17 += 4;
    if ( v88 == v17 )
    {
      v15 = v87 - v17;
      goto LABEL_82;
    }
  }
  if ( (v97 & 1) != 0 )
  {
    v23 = &v98;
    v24 = 3;
  }
  else
  {
    v23 = v98;
    if ( !v99 )
      goto LABEL_29;
    v24 = v99 - 1;
  }
  v25 = v24 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
  v26 = &v23[v25];
  v27 = *v26;
  if ( v22 == *v26 )
    goto LABEL_28;
  v63 = 1;
  while ( v27 != -4 )
  {
    v64 = v63 + 1;
    v25 = v24 & (v63 + v25);
    v26 = &v23[v25];
    v27 = *v26;
    if ( v22 == *v26 )
      goto LABEL_28;
    v63 = v64;
  }
LABEL_29:
  if ( v87 == v17 || (v28 = v17 + 1, v87 == v17 + 1) )
  {
    v86 = v100;
    v36 = &v100[(unsigned int)v101];
    v85 = (__int64 *)(unsigned int)v101;
    goto LABEL_40;
  }
  v29 = v17;
  do
  {
    while ( 1 )
    {
      v30 = *v28;
      v89 = *v28;
      if ( sub_161FC80((__int64)&v90, &v89) )
      {
        *v29++ = v30;
        goto LABEL_33;
      }
      if ( (v97 & 1) != 0 )
      {
        v31 = &v98;
        v32 = 3;
        goto LABEL_37;
      }
      v31 = v98;
      if ( v99 )
        break;
LABEL_33:
      if ( v87 == ++v28 )
        goto LABEL_39;
    }
    v32 = v99 - 1;
LABEL_37:
    v33 = v32 & (((unsigned int)v30 >> 9) ^ ((unsigned int)v30 >> 4));
    v34 = &v31[v33];
    v35 = *v34;
    if ( v30 != *v34 )
    {
      v67 = 1;
      while ( v35 != -4 )
      {
        v68 = v67 + 1;
        v33 = v32 & (v67 + v33);
        v34 = &v31[v33];
        v35 = *v34;
        if ( v30 == *v34 )
          goto LABEL_38;
        v67 = v68;
      }
      goto LABEL_33;
    }
LABEL_38:
    *v34 = -8;
    ++v28;
    ++HIDWORD(v97);
    LODWORD(v97) = (2 * ((unsigned int)v97 >> 1) - 2) | v97 & 1;
  }
  while ( v87 != v28 );
LABEL_39:
  v17 = v29;
  v85 = (__int64 *)(unsigned int)v101;
  v86 = v100;
  v36 = &v100[(unsigned int)v101];
LABEL_40:
  if ( v17 != v36 )
  {
    LODWORD(v101) = v17 - v86;
    v85 = (__int64 *)(unsigned int)v101;
  }
LABEL_42:
  v37 = (__int64 *)(*(_QWORD *)(a1 + 16) & 0xFFFFFFFFFFFFFFF8LL);
  if ( (*(_QWORD *)(a1 + 16) & 4) != 0 )
    v37 = (__int64 *)*v37;
  v38 = sub_1628280(v37, v86, v85);
  if ( v91 != v92 )
    _libc_free((unsigned __int64)v92);
  if ( v100 != (__int64 *)v102 )
    _libc_free((unsigned __int64)v100);
  if ( (v97 & 1) == 0 )
    j___libc_free_0(v98);
  return v38;
}
