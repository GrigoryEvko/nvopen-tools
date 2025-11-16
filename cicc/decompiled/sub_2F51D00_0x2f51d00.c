// Function: sub_2F51D00
// Address: 0x2f51d00
//
__int64 __fastcall sub_2F51D00(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  _QWORD *v5; // r9
  __int64 v7; // r12
  __int64 v8; // rdx
  _QWORD *v9; // r14
  __int64 v10; // r13
  __int64 v11; // rax
  unsigned int *v12; // r10
  unsigned int *v13; // r11
  unsigned __int64 v14; // r9
  unsigned int *v15; // r8
  __int64 v16; // r13
  _QWORD *v17; // r10
  __int64 v18; // rax
  int *v19; // r15
  unsigned __int64 v20; // rax
  int *v21; // r14
  int v22; // ebx
  __int64 v23; // rax
  _QWORD *v24; // rdx
  __int64 v25; // rax
  __int64 v26; // r8
  unsigned __int64 v27; // r8
  unsigned int *v28; // rsi
  __int64 v29; // rax
  __int64 v30; // rcx
  int v31; // eax
  __int64 v32; // rdi
  int v33; // eax
  unsigned int v34; // r11d
  __int64 *v35; // rdx
  __int64 v36; // r9
  __int64 v37; // rax
  char v38; // al
  unsigned int v39; // r12d
  unsigned int *v41; // rdx
  __int64 v42; // rdi
  unsigned int *v43; // r15
  __int64 v44; // r11
  __int64 v45; // r10
  unsigned int v46; // r9d
  __int64 *v47; // rdi
  __int64 v48; // r11
  __int64 v49; // r10
  unsigned int v50; // r9d
  __int64 *v51; // rdi
  __int64 v52; // r11
  __int64 v53; // r11
  unsigned int v54; // r9d
  __int64 *v55; // rdi
  __int64 v56; // r10
  unsigned int v57; // edi
  __int64 *v58; // rdx
  __int64 v59; // r9
  int v60; // edx
  int v61; // edx
  int v62; // r10d
  int v63; // edi
  int v64; // r13d
  int v65; // edi
  int v66; // r13d
  int v67; // edi
  int v68; // r13d
  signed __int64 v69; // rdi
  __int64 v70; // r11
  unsigned int v71; // r15d
  __int64 *v72; // rdi
  __int64 v73; // r9
  int v74; // r10d
  int v75; // edi
  int v76; // r10d
  __int64 v77; // r11
  unsigned int v78; // r15d
  __int64 *v79; // rdi
  __int64 v80; // r9
  __int64 v81; // r11
  unsigned int v82; // r15d
  __int64 *v83; // rdi
  __int64 v84; // r9
  int v85; // edi
  int v86; // r10d
  int v87; // edi
  int v88; // r10d
  unsigned int *v89; // [rsp+8h] [rbp-D8h]
  __int64 v90; // [rsp+10h] [rbp-D0h]
  __int64 v91; // [rsp+18h] [rbp-C8h]
  unsigned int v92; // [rsp+28h] [rbp-B8h]
  __int64 v93; // [rsp+28h] [rbp-B8h]
  unsigned int v94; // [rsp+30h] [rbp-B0h]
  _QWORD *v95; // [rsp+30h] [rbp-B0h]
  unsigned int *v96; // [rsp+38h] [rbp-A8h]
  unsigned int *v97; // [rsp+40h] [rbp-A0h]
  unsigned __int64 v98; // [rsp+48h] [rbp-98h]
  __int64 v99; // [rsp+48h] [rbp-98h]
  __int64 v100; // [rsp+50h] [rbp-90h] BYREF
  __int64 v101; // [rsp+58h] [rbp-88h]
  unsigned __int64 v102[2]; // [rsp+60h] [rbp-80h] BYREF
  _BYTE v103[48]; // [rsp+70h] [rbp-70h] BYREF
  int v104; // [rsp+A0h] [rbp-40h]

  v5 = (_QWORD *)a1;
  v7 = *(_QWORD *)(a1 + 992);
  v102[0] = (unsigned __int64)v103;
  v8 = *(unsigned int *)(v7 + 632);
  v102[1] = 0x600000000LL;
  if ( (_DWORD)v8 )
  {
    sub_2F4CBD0((__int64)v102, v7 + 624, v8, a4, a5, a1);
    v5 = (_QWORD *)a1;
  }
  v9 = v5;
  v92 = 0;
  v104 = *(_DWORD *)(v7 + 688);
  v10 = qword_5023D08;
  while ( 1 )
  {
    v11 = v9[104];
    v12 = *(unsigned int **)(v11 + 88);
    v13 = &v12[*(unsigned int *)(v11 + 96)];
    if ( v12 != v13 )
    {
      v14 = v10;
      v15 = *(unsigned int **)(v11 + 88);
      v16 = a2;
      v17 = v9;
      do
      {
        v18 = *(_QWORD *)(v17[103] + 64LL) + 48LL * *v15;
        v19 = *(int **)v18;
        v20 = *(unsigned int *)(v18 + 8);
        if ( v20 >= v14 )
          goto LABEL_31;
        v21 = &v19[v20];
        for ( v14 -= v20; v21 != v19; ++v19 )
        {
          v22 = *v19;
          v23 = 1LL << *v19;
          v24 = (_QWORD *)(v102[0] + 8LL * ((unsigned int)*v19 >> 6));
          if ( (*v24 & v23) != 0 )
          {
            *v24 &= ~v23;
            v25 = *(unsigned int *)(v16 + 104);
            if ( v25 + 1 > (unsigned __int64)*(unsigned int *)(v16 + 108) )
            {
              v95 = v17;
              v96 = v15;
              v97 = v13;
              v98 = v14;
              sub_C8D5F0(v16 + 96, (const void *)(v16 + 112), v25 + 1, 4u, (__int64)v15, v14);
              v25 = *(unsigned int *)(v16 + 104);
              v17 = v95;
              v15 = v96;
              v13 = v97;
              v14 = v98;
            }
            *(_DWORD *)(*(_QWORD *)(v16 + 96) + 4 * v25) = v22;
            ++*(_DWORD *)(v16 + 104);
          }
        }
        ++v15;
      }
      while ( v13 != v15 );
      a2 = v16;
      v9 = v17;
      v10 = v14;
    }
    v26 = *(unsigned int *)(a2 + 104);
    v94 = v26;
    if ( v92 == (_DWORD)v26 )
    {
      v39 = 1;
      goto LABEL_32;
    }
    v27 = v26 - v92;
    v28 = (unsigned int *)(*(_QWORD *)(a2 + 96) + 4LL * v92);
    if ( *(_DWORD *)a2 )
      break;
    if ( !*(_BYTE *)(v9[124] + 700LL) )
      goto LABEL_23;
    if ( v27 <= 1 )
      goto LABEL_23;
    v29 = v9[101];
    v30 = *(_QWORD *)(v29 + 8);
    v31 = *(_DWORD *)(v29 + 24);
    v91 = *(_QWORD *)(v9[96] + 96LL);
    v32 = *(_QWORD *)(v91 + 8LL * *v28);
    if ( !v31 )
      goto LABEL_23;
    v33 = v31 - 1;
    v34 = v33 & (((unsigned int)v32 >> 9) ^ ((unsigned int)v32 >> 4));
    v35 = (__int64 *)(v30 + 16LL * v34);
    v36 = *v35;
    if ( v32 != *v35 )
    {
      v61 = 1;
      while ( v36 != -4096 )
      {
        v62 = v61 + 1;
        v34 = v33 & (v61 + v34);
        v35 = (__int64 *)(v30 + 16LL * v34);
        v36 = *v35;
        if ( v32 == *v35 )
          goto LABEL_21;
        v61 = v62;
      }
      goto LABEL_23;
    }
LABEL_21:
    v93 = v35[1];
    if ( !v93 || *v28 != *(_DWORD *)(**(_QWORD **)(v93 + 32) + 24LL) )
      goto LABEL_23;
    v41 = v28 + 1;
    v42 = (__int64)(4 * v27 - 4) >> 4;
    v89 = &v28[v27];
    if ( v42 <= 0 )
      goto LABEL_71;
    v43 = v28 + 1;
    v90 = (__int64)&v28[4 * v42 + 1];
    v44 = *(_QWORD *)(v91 + 8LL * v28[1]);
    v99 = v10;
    while ( 1 )
    {
      v57 = v33 & (((unsigned int)v44 >> 9) ^ ((unsigned int)v44 >> 4));
      v58 = (__int64 *)(v30 + 16LL * v57);
      v59 = *v58;
      if ( *v58 != v44 )
      {
        v60 = 1;
        while ( v59 != -4096 )
        {
          v74 = v60 + 1;
          v57 = v33 & (v60 + v57);
          v58 = (__int64 *)(v30 + 16LL * v57);
          v59 = *v58;
          if ( *v58 == v44 )
            goto LABEL_37;
          v60 = v74;
        }
LABEL_49:
        v10 = v99;
        v41 = v43;
        goto LABEL_50;
      }
LABEL_37:
      if ( v93 != v58[1] )
        goto LABEL_49;
      v41 = v43 + 1;
      v45 = *(_QWORD *)(v91 + 8LL * v43[1]);
      v46 = v33 & (((unsigned int)v45 >> 4) ^ ((unsigned int)v45 >> 9));
      v47 = (__int64 *)(v30 + 16LL * v46);
      v48 = *v47;
      if ( v45 != *v47 )
      {
        v63 = 1;
        while ( v48 != -4096 )
        {
          v64 = v63 + 1;
          v46 = v33 & (v63 + v46);
          v47 = (__int64 *)(v30 + 16LL * v46);
          v48 = *v47;
          if ( v45 == *v47 )
            goto LABEL_39;
          v63 = v64;
        }
        goto LABEL_56;
      }
LABEL_39:
      if ( v93 != v47[1] )
        goto LABEL_56;
      v41 = v43 + 2;
      v49 = *(_QWORD *)(v91 + 8LL * v43[2]);
      v50 = v33 & (((unsigned int)v49 >> 4) ^ ((unsigned int)v49 >> 9));
      v51 = (__int64 *)(v30 + 16LL * v50);
      v52 = *v51;
      if ( v49 != *v51 )
      {
        v65 = 1;
        while ( v52 != -4096 )
        {
          v66 = v65 + 1;
          v50 = v33 & (v65 + v50);
          v51 = (__int64 *)(v30 + 16LL * v50);
          v52 = *v51;
          if ( v49 == *v51 )
            goto LABEL_41;
          v65 = v66;
        }
        goto LABEL_56;
      }
LABEL_41:
      if ( v93 != v51[1] )
        goto LABEL_56;
      v41 = v43 + 3;
      v53 = *(_QWORD *)(v91 + 8LL * v43[3]);
      v54 = v33 & (((unsigned int)v53 >> 4) ^ ((unsigned int)v53 >> 9));
      v55 = (__int64 *)(v30 + 16LL * v54);
      v56 = *v55;
      if ( v53 != *v55 )
      {
        v67 = 1;
        while ( v56 != -4096 )
        {
          v68 = v67 + 1;
          v54 = v33 & (v67 + v54);
          v55 = (__int64 *)(v30 + 16LL * v54);
          v56 = *v55;
          if ( *v55 == v53 )
            goto LABEL_43;
          v67 = v68;
        }
LABEL_56:
        v10 = v99;
        goto LABEL_50;
      }
LABEL_43:
      if ( v93 != v55[1] )
        goto LABEL_56;
      v43 += 4;
      if ( (unsigned int *)v90 == v43 )
        break;
      v44 = *(_QWORD *)(v91 + 8LL * *v43);
    }
    v10 = v99;
    v41 = (unsigned int *)v90;
LABEL_71:
    v69 = (char *)v89 - (char *)v41;
    if ( (char *)v89 - (char *)v41 == 8 )
      goto LABEL_86;
    if ( v69 == 12 )
    {
      v77 = *(_QWORD *)(v91 + 8LL * *v41);
      v78 = v33 & (((unsigned int)v77 >> 9) ^ ((unsigned int)v77 >> 4));
      v79 = (__int64 *)(v30 + 16LL * v78);
      v80 = *v79;
      if ( *v79 == v77 )
      {
LABEL_84:
        if ( v93 == v79[1] )
        {
          ++v41;
LABEL_86:
          v81 = *(_QWORD *)(v91 + 8LL * *v41);
          v82 = v33 & (((unsigned int)v81 >> 9) ^ ((unsigned int)v81 >> 4));
          v83 = (__int64 *)(v30 + 16LL * v82);
          v84 = *v83;
          if ( *v83 == v81 )
          {
LABEL_87:
            if ( v93 == v83[1] )
            {
              ++v41;
LABEL_74:
              v70 = *(_QWORD *)(v91 + 8LL * *v41);
              v71 = v33 & (((unsigned int)v70 >> 9) ^ ((unsigned int)v70 >> 4));
              v72 = (__int64 *)(v30 + 16LL * v71);
              v73 = *v72;
              if ( v70 == *v72 )
              {
LABEL_75:
                if ( v93 == v72[1] )
                  goto LABEL_51;
              }
              else
              {
                v75 = 1;
                while ( v73 != -4096 )
                {
                  v76 = v75 + 1;
                  v71 = v33 & (v75 + v71);
                  v72 = (__int64 *)(v30 + 16LL * v71);
                  v73 = *v72;
                  if ( v70 == *v72 )
                    goto LABEL_75;
                  v75 = v76;
                }
              }
            }
          }
          else
          {
            v87 = 1;
            while ( v84 != -4096 )
            {
              v88 = v87 + 1;
              v82 = v33 & (v87 + v82);
              v83 = (__int64 *)(v30 + 16LL * v82);
              v84 = *v83;
              if ( v81 == *v83 )
                goto LABEL_87;
              v87 = v88;
            }
          }
        }
      }
      else
      {
        v85 = 1;
        while ( v80 != -4096 )
        {
          v86 = v85 + 1;
          v78 = v33 & (v85 + v78);
          v79 = (__int64 *)(v30 + 16LL * v78);
          v80 = *v79;
          if ( v77 == *v79 )
            goto LABEL_84;
          v85 = v86;
        }
      }
LABEL_50:
      if ( v89 == v41 )
        goto LABEL_51;
LABEL_23:
      sub_2FAF7F0(v9[104], v28, v27, 1);
LABEL_24:
      v92 = *(_DWORD *)(a2 + 104);
      goto LABEL_25;
    }
    if ( v69 == 4 )
      goto LABEL_74;
LABEL_51:
    v92 = v94;
LABEL_25:
    sub_2FAFF40(v9[104]);
  }
  v101 = 0;
  v37 = *(_QWORD *)(a2 + 8);
  v100 = v37;
  if ( v37 )
    ++*(_DWORD *)(v37 + 8);
  v38 = sub_2F51790(v9, &v100, v28, v27);
  v101 = 0;
  if ( v100 )
    --*(_DWORD *)(v100 + 8);
  if ( v38 )
    goto LABEL_24;
LABEL_31:
  v39 = 0;
LABEL_32:
  if ( (_BYTE *)v102[0] != v103 )
    _libc_free(v102[0]);
  return v39;
}
