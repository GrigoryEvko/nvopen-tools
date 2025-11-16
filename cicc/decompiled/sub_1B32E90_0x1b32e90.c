// Function: sub_1B32E90
// Address: 0x1b32e90
//
__int64 __fastcall sub_1B32E90(__int64 *a1, __int64 *a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  __int64 *v5; // rbx
  __int64 v6; // rcx
  int v7; // eax
  int v8; // eax
  __int64 v9; // r15
  __int64 v10; // rsi
  __int64 *v11; // rdx
  __int64 v12; // r12
  __int64 *v13; // rdi
  unsigned int v14; // r11d
  __int64 v15; // r9
  __int64 *v16; // r13
  __int64 v17; // rdi
  __int64 v18; // r14
  unsigned int v19; // r8d
  __int64 v20; // r11
  __int64 *v21; // r13
  __int64 v22; // r14
  __int64 *v23; // r9
  __int64 v24; // r10
  __int64 v25; // r11
  __int64 v26; // rdx
  __int64 *v27; // r9
  __int64 v28; // rdi
  int v29; // ebx
  int v30; // r12d
  unsigned int v31; // ecx
  __int64 *v32; // rax
  __int64 v33; // r8
  unsigned int v34; // ecx
  __int64 v35; // r14
  __int64 v36; // r13
  __int64 *v37; // rax
  __int64 *v38; // rsi
  int v39; // r10d
  unsigned int v40; // ecx
  __int64 *v41; // rax
  __int64 v42; // rbx
  unsigned int v43; // r10d
  unsigned int v44; // ecx
  __int64 *v45; // rax
  __int64 v46; // rbx
  int v47; // eax
  int v48; // eax
  __int64 v49; // r10
  int v50; // eax
  int v51; // r10d
  __int64 v52; // rax
  int v53; // eax
  __int64 v54; // r10
  int v55; // r8d
  __int64 *v56; // r12
  __int64 *v57; // rdx
  __int64 v58; // rbx
  __int64 v59; // r12
  __int64 v60; // r8
  __int64 *v61; // rbx
  __int64 v62; // rcx
  int v63; // r10d
  __int64 v64; // r10
  int v65; // r9d
  unsigned int v66; // ecx
  int v67; // r8d
  unsigned int v68; // r10d
  __int64 v69; // r8
  int v70; // edx
  int v71; // r8d
  unsigned int v72; // r9d
  unsigned int v73; // r10d
  __int64 v74; // r8
  int v75; // edi
  unsigned int v76; // r8d
  int v77; // edx
  int v78; // r10d
  __int64 v79; // r8
  __int64 *v80; // r8
  int i; // r8d
  unsigned int v82; // r8d
  int v83; // edx
  __int64 v84; // r13
  unsigned int v85; // edx
  int v86; // r8d
  int v87; // r9d
  int v88; // ecx
  __int64 v89; // rdx
  int v90; // r10d
  __int64 v91; // r9
  int v92; // r10d
  int v93; // r11d
  int v94; // r10d
  __int64 *v95; // r8
  __int64 *v96; // [rsp+0h] [rbp-70h]
  int v97; // [rsp+8h] [rbp-68h]
  unsigned int v98; // [rsp+Ch] [rbp-64h]
  int v99; // [rsp+Ch] [rbp-64h]
  int v100; // [rsp+Ch] [rbp-64h]
  __int64 v101; // [rsp+10h] [rbp-60h]
  __int64 *v102; // [rsp+18h] [rbp-58h]
  __int64 *v105; // [rsp+30h] [rbp-40h]
  int v106; // [rsp+30h] [rbp-40h]
  unsigned int v107; // [rsp+30h] [rbp-40h]
  unsigned int v108; // [rsp+30h] [rbp-40h]
  __int64 *v109; // [rsp+38h] [rbp-38h]
  unsigned int v110; // [rsp+38h] [rbp-38h]

  result = (char *)a2 - (char *)a1;
  v102 = a2;
  v101 = a3;
  if ( (char *)a2 - (char *)a1 <= 128 )
    return result;
  if ( !a3 )
  {
    v109 = a2;
    goto LABEL_53;
  }
  v96 = a1 + 2;
  while ( 2 )
  {
    --v101;
    v5 = &a1[result >> 4];
    v6 = *v5;
    v7 = *(_DWORD *)(a4 + 920);
    if ( !v7 )
    {
      v24 = *a1;
      goto LABEL_11;
    }
    v8 = v7 - 1;
    v9 = a1[1];
    v10 = *(_QWORD *)(a4 + 904);
    v11 = (__int64 *)(v10 + 16LL * (v8 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4))));
    v12 = *v11;
    v13 = v11;
    if ( v9 == *v11 )
    {
LABEL_6:
      v14 = *((_DWORD *)v13 + 2);
    }
    else
    {
      v73 = v8 & (((unsigned int)a1[1] >> 9) ^ ((unsigned int)v9 >> 4));
      v74 = *v11;
      v75 = 1;
      while ( v74 != -8 )
      {
        v87 = v75 + 1;
        v73 = v8 & (v75 + v73);
        v13 = (__int64 *)(v10 + 16LL * v73);
        v74 = *v13;
        if ( v9 == *v13 )
          goto LABEL_6;
        v75 = v87;
      }
      v14 = 0;
    }
    v15 = v8 & (((unsigned int)v6 >> 4) ^ ((unsigned int)v6 >> 9));
    v16 = (__int64 *)(v10 + 16 * v15);
    v17 = *(v102 - 1);
    v18 = *v16;
    if ( v6 == *v16 )
    {
      v19 = *((_DWORD *)v16 + 2);
      if ( v19 > v14 )
      {
LABEL_9:
        v20 = v8 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
        v21 = (__int64 *)(v10 + 16 * v20);
        v22 = *v21;
        v23 = v21;
        if ( v17 == *v21 )
        {
LABEL_10:
          v24 = *a1;
          if ( v19 < *((_DWORD *)v23 + 2) )
            goto LABEL_11;
        }
        else
        {
          v108 = v8 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
          v64 = *v21;
          v65 = 1;
          while ( v64 != -8 )
          {
            v90 = v65 + 1;
            v91 = v8 & (v108 + v65);
            v100 = v90;
            v108 = v91;
            v23 = (__int64 *)(v10 + 16 * v91);
            v64 = *v23;
            if ( v17 == *v23 )
              goto LABEL_10;
            v65 = v100;
          }
          v24 = *a1;
        }
        if ( v9 == v12 )
        {
LABEL_65:
          v66 = *((_DWORD *)v11 + 2);
        }
        else
        {
          v82 = v8 & (((unsigned int)a1[1] >> 9) ^ ((unsigned int)v9 >> 4));
          v83 = 1;
          while ( v12 != -8 )
          {
            v88 = v83 + 1;
            v89 = v8 & (v82 + v83);
            v82 = v89;
            v11 = (__int64 *)(v10 + 16 * v89);
            v12 = *v11;
            if ( v9 == *v11 )
              goto LABEL_65;
            v83 = v88;
          }
          v66 = 0;
        }
        v67 = 1;
        if ( v17 == v22 )
        {
LABEL_67:
          v26 = v24;
          if ( *((_DWORD *)v21 + 2) > v66 )
            goto LABEL_68;
        }
        else
        {
          while ( v22 != -8 )
          {
            LODWORD(v20) = v8 & (v67 + v20);
            v21 = (__int64 *)(v10 + 16LL * (unsigned int)v20);
            v22 = *v21;
            if ( v17 == *v21 )
              goto LABEL_67;
            ++v67;
          }
        }
        v25 = v24;
        *a1 = v9;
        a1[1] = v24;
        v26 = *(v102 - 1);
        goto LABEL_12;
      }
    }
    else
    {
      v107 = v8 & (((unsigned int)v6 >> 4) ^ ((unsigned int)v6 >> 9));
      v54 = *v16;
      v55 = 1;
      while ( v54 != -8 )
      {
        v78 = v55 + 1;
        v79 = v8 & (v107 + v55);
        v99 = v78;
        v107 = v79;
        v80 = (__int64 *)(v10 + 16 * v79);
        v54 = *v80;
        if ( v6 == *v80 )
        {
          if ( *((_DWORD *)v80 + 2) <= v14 )
            break;
          for ( i = 1; ; i = v94 )
          {
            if ( v18 == -8 )
            {
              v19 = 0;
              goto LABEL_9;
            }
            v94 = i + 1;
            LODWORD(v15) = v8 & (v15 + i);
            v95 = (__int64 *)(v10 + 16LL * (unsigned int)v15);
            v18 = *v95;
            if ( v6 == *v95 )
              break;
          }
          v19 = *((_DWORD *)v95 + 2);
          goto LABEL_9;
        }
        v55 = v99;
      }
    }
    if ( v9 == v12 )
    {
LABEL_48:
      v110 = *((_DWORD *)v11 + 2);
    }
    else
    {
      v76 = v8 & (((unsigned int)a1[1] >> 9) ^ ((unsigned int)v9 >> 4));
      v77 = 1;
      while ( v12 != -8 )
      {
        v92 = v77 + 1;
        v76 = v8 & (v77 + v76);
        v11 = (__int64 *)(v10 + 16LL * v76);
        v12 = *v11;
        if ( v9 == *v11 )
          goto LABEL_48;
        v77 = v92;
      }
      v110 = 0;
    }
    v56 = (__int64 *)(v10 + 16LL * (v8 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4))));
    v57 = v56;
    if ( *v56 == v17 )
    {
LABEL_50:
      v25 = *a1;
      v24 = *a1;
      if ( *((_DWORD *)v57 + 2) > v110 )
      {
        *a1 = v9;
        a1[1] = v25;
        v26 = *(v102 - 1);
        goto LABEL_12;
      }
    }
    else
    {
      v68 = v8 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v69 = *v56;
      v70 = 1;
      while ( v69 != -8 )
      {
        v93 = v70 + 1;
        v68 = v8 & (v70 + v68);
        v57 = (__int64 *)(v10 + 16LL * v68);
        v69 = *v57;
        if ( v17 == *v57 )
          goto LABEL_50;
        v70 = v93;
      }
      v24 = *a1;
    }
    v71 = 1;
    if ( v6 == v18 )
    {
LABEL_73:
      v72 = *((_DWORD *)v16 + 2);
    }
    else
    {
      while ( v18 != -8 )
      {
        LODWORD(v15) = v8 & (v71 + v15);
        v16 = (__int64 *)(v10 + 16LL * (unsigned int)v15);
        v18 = *v16;
        if ( v6 == *v16 )
          goto LABEL_73;
        ++v71;
      }
      v72 = 0;
    }
    if ( *v56 != v17 )
    {
      v84 = *v56;
      v85 = v8 & (((unsigned int)v17 >> 9) ^ ((unsigned int)v17 >> 4));
      v86 = 1;
      while ( v84 != -8 )
      {
        v85 = v8 & (v86 + v85);
        v56 = (__int64 *)(v10 + 16LL * v85);
        v84 = *v56;
        if ( v17 == *v56 )
          goto LABEL_75;
        ++v86;
      }
      goto LABEL_11;
    }
LABEL_75:
    v26 = v24;
    if ( *((_DWORD *)v56 + 2) > v72 )
    {
LABEL_68:
      *a1 = v17;
      *(v102 - 1) = v26;
      v9 = *a1;
      v25 = a1[1];
      goto LABEL_12;
    }
LABEL_11:
    *a1 = v6;
    *v5 = v24;
    v9 = *a1;
    v25 = a1[1];
    v26 = *(v102 - 1);
LABEL_12:
    v27 = v96;
    v28 = *(_QWORD *)(a4 + 904);
    v29 = *(_DWORD *)(a4 + 920);
    v105 = v102 - 1;
    while ( 1 )
    {
      v38 = v105;
      v109 = v27 - 1;
      if ( !v29 )
        break;
      v30 = v29 - 1;
      v31 = (v29 - 1) & (((unsigned int)v25 >> 9) ^ ((unsigned int)v25 >> 4));
      v32 = (__int64 *)(v28 + 16LL * v31);
      v33 = *v32;
      if ( v25 == *v32 )
      {
LABEL_14:
        v34 = *((_DWORD *)v32 + 2);
      }
      else
      {
        v53 = 1;
        while ( v33 != -8 )
        {
          v63 = v53 + 1;
          v31 = v30 & (v53 + v31);
          v32 = (__int64 *)(v28 + 16LL * v31);
          v33 = *v32;
          if ( v25 == *v32 )
            goto LABEL_14;
          v53 = v63;
        }
        v34 = 0;
      }
      v35 = v30 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
      v36 = *(_QWORD *)(v28 + 16 * v35);
      v37 = (__int64 *)(v28 + 16 * v35);
      if ( v36 != v9 )
      {
        v98 = v30 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
        v49 = *(_QWORD *)(v28 + 16 * v35);
        v50 = 1;
        while ( v49 != -8 )
        {
          v51 = v50 + 1;
          v52 = v30 & (v98 + v50);
          v97 = v51;
          v98 = v52;
          v37 = (__int64 *)(v28 + 16 * v52);
          v49 = *v37;
          if ( v9 == *v37 )
            goto LABEL_16;
          v50 = v97;
        }
        while ( 1 )
        {
LABEL_26:
          v41 = (__int64 *)(v28 + 16 * v35);
          if ( v36 == v9 )
          {
LABEL_22:
            v43 = *((_DWORD *)v41 + 2);
          }
          else
          {
            v42 = *(_QWORD *)(v28 + 16 * v35);
            v40 = v30 & (((unsigned int)v9 >> 9) ^ ((unsigned int)v9 >> 4));
            v47 = 1;
            while ( v42 != -8 )
            {
              v39 = v47 + 1;
              v40 = v30 & (v47 + v40);
              v41 = (__int64 *)(v28 + 16LL * v40);
              v42 = *v41;
              if ( v9 == *v41 )
                goto LABEL_22;
              v47 = v39;
            }
            v43 = 0;
          }
          v44 = v30 & (((unsigned int)v26 >> 9) ^ ((unsigned int)v26 >> 4));
          v45 = (__int64 *)(v28 + 16LL * v44);
          v46 = *v45;
          if ( v26 != *v45 )
            break;
LABEL_24:
          if ( v43 >= *((_DWORD *)v45 + 2) )
            goto LABEL_19;
          v26 = *--v38;
        }
        v48 = 1;
        while ( v46 != -8 )
        {
          v44 = v30 & (v48 + v44);
          v106 = v48 + 1;
          v45 = (__int64 *)(v28 + 16LL * v44);
          v46 = *v45;
          if ( v26 == *v45 )
            goto LABEL_24;
          v48 = v106;
        }
        break;
      }
LABEL_16:
      if ( *((_DWORD *)v37 + 2) <= v34 )
        goto LABEL_26;
LABEL_17:
      v25 = *v27++;
    }
LABEL_19:
    if ( v38 > v109 )
    {
      *(v27 - 1) = v26;
      *v38 = v25;
      v26 = *(v38 - 1);
      v9 = *a1;
      v28 = *(_QWORD *)(a4 + 904);
      v29 = *(_DWORD *)(a4 + 920);
      v105 = v38 - 1;
      goto LABEL_17;
    }
    sub_1B32E90(v109, v102, v101, a4);
    result = (char *)v109 - (char *)a1;
    if ( (char *)v109 - (char *)a1 > 128 )
    {
      if ( v101 )
      {
        v102 = v109;
        continue;
      }
LABEL_53:
      v58 = result >> 3;
      v59 = ((result >> 3) - 2) >> 1;
      sub_1B321D0((__int64)a1, v59, result >> 3, a1[v59], a4);
      do
      {
        --v59;
        sub_1B321D0((__int64)a1, v59, v58, a1[v59], v60);
      }
      while ( v59 );
      v61 = v109;
      do
      {
        v62 = *--v61;
        *v61 = *a1;
        result = sub_1B321D0((__int64)a1, 0, v61 - a1, v62, v60);
      }
      while ( (char *)v61 - (char *)a1 > 8 );
    }
    return result;
  }
}
