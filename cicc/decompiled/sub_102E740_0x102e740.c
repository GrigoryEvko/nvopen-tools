// Function: sub_102E740
// Address: 0x102e740
//
__int64 __fastcall sub_102E740(__int64 a1)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rax
  _BYTE *v4; // rdi
  __int64 v5; // rbx
  char v6; // cl
  char v7; // r8
  __int64 v8; // rsi
  int v9; // r10d
  unsigned int v10; // r9d
  _QWORD *v11; // rax
  _BYTE *v12; // rdx
  __int64 v13; // rdx
  __int64 v14; // r15
  __int64 *v15; // r12
  __int64 *v16; // r9
  __int64 v17; // r8
  int v18; // esi
  unsigned int v19; // edx
  _QWORD *v20; // rax
  __int64 v21; // r10
  __int64 v22; // r14
  int v23; // ecx
  unsigned int v24; // esi
  unsigned int v25; // eax
  _QWORD *v26; // rdi
  int v27; // edx
  unsigned int v28; // r8d
  __int64 v29; // r15
  char v30; // al
  __int64 v32; // rax
  int v33; // r11d
  __int64 v34; // r8
  __int64 v35; // r9
  __int64 *v36; // r12
  __int64 *v37; // r9
  __int64 v38; // r8
  int v39; // esi
  unsigned int v40; // edx
  _QWORD *v41; // rax
  __int64 v42; // r10
  __int64 v43; // r14
  __int64 v44; // r15
  char v45; // cl
  unsigned int v46; // esi
  int v47; // eax
  __int64 v48; // rdx
  int v49; // ecx
  unsigned int v50; // eax
  __int64 v51; // rsi
  int v52; // edx
  __int64 v53; // rcx
  int v54; // edx
  unsigned int v55; // eax
  __int64 v56; // rsi
  int v57; // r10d
  _QWORD *v58; // r8
  __int64 v59; // rax
  __int64 v60; // rbx
  unsigned int v61; // eax
  _QWORD *v62; // rdi
  int v63; // edx
  unsigned int v64; // r8d
  __int64 v65; // rbx
  int v66; // r11d
  __int64 v67; // rax
  int v68; // eax
  int v69; // edx
  __int64 v70; // rcx
  int v71; // edx
  unsigned int v72; // eax
  __int64 v73; // rsi
  int v74; // edx
  __int64 v75; // rcx
  int v76; // edx
  unsigned int v77; // eax
  __int64 v78; // rsi
  int v79; // r10d
  _QWORD *v80; // r8
  int v81; // r10d
  int v82; // r11d
  int v83; // r10d
  __int64 *v84; // [rsp+8h] [rbp-38h]
  __int64 *v85; // [rsp+8h] [rbp-38h]
  __int64 *v86; // [rsp+8h] [rbp-38h]
  __int64 *v87; // [rsp+8h] [rbp-38h]

  v2 = **(_QWORD **)(a1 + 16);
  *(_WORD *)(a1 + 24) = 0;
  v3 = v2 & 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(a1 + 16) = v3;
  v4 = 0;
  v5 = *(_QWORD *)a1;
  if ( v3 )
    v4 = (_BYTE *)(v3 - 24);
  *(_QWORD *)(a1 + 32) = v4;
  v6 = *(_BYTE *)(v5 + 8);
  v7 = v6 & 1;
  if ( (v6 & 1) != 0 )
  {
    v8 = v5 + 16;
    v9 = 3;
  }
  else
  {
    v32 = *(unsigned int *)(v5 + 24);
    v8 = *(_QWORD *)(v5 + 16);
    if ( !(_DWORD)v32 )
      goto LABEL_80;
    v9 = v32 - 1;
  }
  v10 = v9 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v11 = (_QWORD *)(v8 + 16LL * v10);
  v12 = (_BYTE *)*v11;
  if ( v4 == (_BYTE *)*v11 )
    goto LABEL_6;
  v68 = 1;
  while ( v12 != (_BYTE *)-4096LL )
  {
    v82 = v68 + 1;
    v10 = v9 & (v68 + v10);
    v11 = (_QWORD *)(v8 + 16LL * v10);
    v12 = (_BYTE *)*v11;
    if ( v4 == (_BYTE *)*v11 )
      goto LABEL_6;
    v68 = v82;
  }
  if ( v7 )
  {
    v67 = 64;
    goto LABEL_81;
  }
  v32 = *(unsigned int *)(v5 + 24);
LABEL_80:
  v67 = 16 * v32;
LABEL_81:
  v11 = (_QWORD *)(v8 + v67);
LABEL_6:
  v13 = 64;
  if ( !v7 )
    v13 = 16LL * *(unsigned int *)(v5 + 24);
  if ( v11 != (_QWORD *)(v8 + v13) )
  {
    v14 = v11[1];
    v15 = *(__int64 **)(a1 + 40);
    v16 = &v15[*(unsigned int *)(a1 + 48)];
    if ( v16 == v15 )
    {
LABEL_25:
      *(_DWORD *)(a1 + 48) = 0;
      if ( v14 )
      {
        v29 = v14 + 24;
        v30 = 0;
      }
      else
      {
        v29 = *(_QWORD *)(*(_QWORD *)(a1 + 8) + 56LL);
        v30 = 1;
      }
      *(_QWORD *)(a1 + 16) = v29;
      *(_BYTE *)(a1 + 24) = v30;
      *(_BYTE *)(a1 + 25) = 0;
      return a1;
    }
    while ( 1 )
    {
      v22 = *v15;
      v23 = v6 & 1;
      if ( v23 )
      {
        v17 = v5 + 16;
        v18 = 3;
      }
      else
      {
        v24 = *(_DWORD *)(v5 + 24);
        v17 = *(_QWORD *)(v5 + 16);
        if ( !v24 )
        {
          v25 = *(_DWORD *)(v5 + 8);
          ++*(_QWORD *)v5;
          v26 = 0;
          v27 = (v25 >> 1) + 1;
LABEL_19:
          v28 = 3 * v24;
          goto LABEL_20;
        }
        v18 = v24 - 1;
      }
      v19 = v18 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v20 = (_QWORD *)(v17 + 16LL * v19);
      v21 = *v20;
      if ( v22 == *v20 )
      {
LABEL_13:
        ++v15;
        v20[1] = v14;
        if ( v16 == v15 )
          goto LABEL_25;
        goto LABEL_14;
      }
      v33 = 1;
      v26 = 0;
      while ( v21 != -4096 )
      {
        if ( v21 == -8192 && !v26 )
          v26 = v20;
        v19 = v18 & (v33 + v19);
        v20 = (_QWORD *)(v17 + 16LL * v19);
        v21 = *v20;
        if ( v22 == *v20 )
          goto LABEL_13;
        ++v33;
      }
      v28 = 12;
      v24 = 4;
      if ( !v26 )
        v26 = v20;
      v25 = *(_DWORD *)(v5 + 8);
      ++*(_QWORD *)v5;
      v27 = (v25 >> 1) + 1;
      if ( !(_BYTE)v23 )
      {
        v24 = *(_DWORD *)(v5 + 24);
        goto LABEL_19;
      }
LABEL_20:
      if ( 4 * v27 >= v28 )
      {
        v84 = v16;
        sub_102E030(v5, 2 * v24);
        v16 = v84;
        if ( (*(_BYTE *)(v5 + 8) & 1) != 0 )
        {
          v48 = v5 + 16;
          v49 = 3;
        }
        else
        {
          v47 = *(_DWORD *)(v5 + 24);
          v48 = *(_QWORD *)(v5 + 16);
          if ( !v47 )
            goto LABEL_137;
          v49 = v47 - 1;
        }
        v50 = v49 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
        v26 = (_QWORD *)(v48 + 16LL * v50);
        v51 = *v26;
        if ( v22 == *v26 )
          goto LABEL_51;
        v81 = 1;
        v58 = 0;
        while ( v51 != -4096 )
        {
          if ( v51 == -8192 && !v58 )
            v58 = v26;
          v50 = v49 & (v81 + v50);
          v26 = (_QWORD *)(v48 + 16LL * v50);
          v51 = *v26;
          if ( v22 == *v26 )
            goto LABEL_51;
          ++v81;
        }
        goto LABEL_58;
      }
      if ( v24 - *(_DWORD *)(v5 + 12) - v27 > v24 >> 3 )
        goto LABEL_22;
      v85 = v16;
      sub_102E030(v5, v24);
      v16 = v85;
      if ( (*(_BYTE *)(v5 + 8) & 1) != 0 )
      {
        v53 = v5 + 16;
        v54 = 3;
      }
      else
      {
        v52 = *(_DWORD *)(v5 + 24);
        v53 = *(_QWORD *)(v5 + 16);
        if ( !v52 )
          goto LABEL_137;
        v54 = v52 - 1;
      }
      v55 = v54 & (((unsigned int)v22 >> 9) ^ ((unsigned int)v22 >> 4));
      v26 = (_QWORD *)(v53 + 16LL * v55);
      v56 = *v26;
      if ( v22 != *v26 )
      {
        v57 = 1;
        v58 = 0;
        while ( v56 != -4096 )
        {
          if ( !v58 && v56 == -8192 )
            v58 = v26;
          v55 = v54 & (v57 + v55);
          v26 = (_QWORD *)(v53 + 16LL * v55);
          v56 = *v26;
          if ( v22 == *v26 )
            goto LABEL_51;
          ++v57;
        }
LABEL_58:
        if ( v58 )
          v26 = v58;
      }
LABEL_51:
      v25 = *(_DWORD *)(v5 + 8);
LABEL_22:
      *(_DWORD *)(v5 + 8) = (2 * (v25 >> 1) + 2) | v25 & 1;
      if ( *v26 != -4096 )
        --*(_DWORD *)(v5 + 12);
      ++v15;
      *v26 = v22;
      v26[1] = 0;
      v26[1] = v14;
      if ( v16 == v15 )
        goto LABEL_25;
LABEL_14:
      v5 = *(_QWORD *)a1;
      v6 = *(_BYTE *)(*(_QWORD *)a1 + 8LL);
    }
  }
  if ( (unsigned __int8)sub_102A860(v4) )
  {
    v36 = *(__int64 **)(a1 + 40);
    v37 = &v36[*(unsigned int *)(a1 + 48)];
    if ( v36 == v37 )
    {
LABEL_72:
      *(_DWORD *)(a1 + 48) = 0;
      v65 = *(_QWORD *)(a1 + 32);
      *sub_102E450(*(_QWORD *)a1, (__int64 *)(a1 + 32)) = v65;
      return a1;
    }
    while ( 1 )
    {
      v5 = *(_QWORD *)a1;
      v43 = *v36;
      v44 = *(_QWORD *)(a1 + 32);
      v45 = *(_BYTE *)(*(_QWORD *)a1 + 8LL) & 1;
      if ( v45 )
      {
        v38 = v5 + 16;
        v39 = 3;
      }
      else
      {
        v46 = *(_DWORD *)(v5 + 24);
        v38 = *(_QWORD *)(v5 + 16);
        if ( !v46 )
        {
          v61 = *(_DWORD *)(v5 + 8);
          ++*(_QWORD *)v5;
          v62 = 0;
          v63 = (v61 >> 1) + 1;
LABEL_66:
          v64 = 3 * v46;
          goto LABEL_67;
        }
        v39 = v46 - 1;
      }
      v40 = v39 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
      v41 = (_QWORD *)(v38 + 16LL * v40);
      v42 = *v41;
      if ( v43 == *v41 )
      {
LABEL_43:
        ++v36;
        v41[1] = v44;
        if ( v37 == v36 )
          goto LABEL_72;
      }
      else
      {
        v66 = 1;
        v62 = 0;
        while ( v42 != -4096 )
        {
          if ( v42 == -8192 && !v62 )
            v62 = v41;
          v40 = v39 & (v66 + v40);
          v41 = (_QWORD *)(v38 + 16LL * v40);
          v42 = *v41;
          if ( v43 == *v41 )
            goto LABEL_43;
          ++v66;
        }
        v64 = 12;
        v46 = 4;
        if ( !v62 )
          v62 = v41;
        v61 = *(_DWORD *)(v5 + 8);
        ++*(_QWORD *)v5;
        v63 = (v61 >> 1) + 1;
        if ( !v45 )
        {
          v46 = *(_DWORD *)(v5 + 24);
          goto LABEL_66;
        }
LABEL_67:
        if ( v64 <= 4 * v63 )
        {
          v86 = v37;
          sub_102E030(v5, 2 * v46);
          v37 = v86;
          if ( (*(_BYTE *)(v5 + 8) & 1) != 0 )
          {
            v70 = v5 + 16;
            v71 = 3;
          }
          else
          {
            v69 = *(_DWORD *)(v5 + 24);
            v70 = *(_QWORD *)(v5 + 16);
            if ( !v69 )
              goto LABEL_137;
            v71 = v69 - 1;
          }
          v72 = v71 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
          v62 = (_QWORD *)(v70 + 16LL * v72);
          v73 = *v62;
          if ( v43 != *v62 )
          {
            v83 = 1;
            v80 = 0;
            while ( v73 != -4096 )
            {
              if ( v73 == -8192 && !v80 )
                v80 = v62;
              v72 = v71 & (v83 + v72);
              v62 = (_QWORD *)(v70 + 16LL * v72);
              v73 = *v62;
              if ( v43 == *v62 )
                goto LABEL_90;
              ++v83;
            }
LABEL_97:
            if ( v80 )
              v62 = v80;
          }
LABEL_90:
          v61 = *(_DWORD *)(v5 + 8);
          goto LABEL_69;
        }
        if ( v46 - *(_DWORD *)(v5 + 12) - v63 <= v46 >> 3 )
        {
          v87 = v37;
          sub_102E030(v5, v46);
          v37 = v87;
          if ( (*(_BYTE *)(v5 + 8) & 1) != 0 )
          {
            v75 = v5 + 16;
            v76 = 3;
          }
          else
          {
            v74 = *(_DWORD *)(v5 + 24);
            v75 = *(_QWORD *)(v5 + 16);
            if ( !v74 )
            {
LABEL_137:
              *(_DWORD *)(v5 + 8) = (2 * (*(_DWORD *)(v5 + 8) >> 1) + 2) | *(_DWORD *)(v5 + 8) & 1;
              BUG();
            }
            v76 = v74 - 1;
          }
          v77 = v76 & (((unsigned int)v43 >> 9) ^ ((unsigned int)v43 >> 4));
          v62 = (_QWORD *)(v75 + 16LL * v77);
          v78 = *v62;
          if ( v43 != *v62 )
          {
            v79 = 1;
            v80 = 0;
            while ( v78 != -4096 )
            {
              if ( v78 == -8192 && !v80 )
                v80 = v62;
              v77 = v76 & (v79 + v77);
              v62 = (_QWORD *)(v75 + 16LL * v77);
              v78 = *v62;
              if ( v43 == *v62 )
                goto LABEL_90;
              ++v79;
            }
            goto LABEL_97;
          }
          goto LABEL_90;
        }
LABEL_69:
        *(_DWORD *)(v5 + 8) = (2 * (v61 >> 1) + 2) | v61 & 1;
        if ( *v62 != -4096 )
          --*(_DWORD *)(v5 + 12);
        ++v36;
        *v62 = v43;
        v62[1] = 0;
        v62[1] = v44;
        if ( v37 == v36 )
          goto LABEL_72;
      }
    }
  }
  v59 = *(unsigned int *)(a1 + 48);
  v60 = *(_QWORD *)(a1 + 32);
  if ( v59 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 52) )
  {
    sub_C8D5F0(a1 + 40, (const void *)(a1 + 56), v59 + 1, 8u, v34, v35);
    v59 = *(unsigned int *)(a1 + 48);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 40) + 8 * v59) = v60;
  ++*(_DWORD *)(a1 + 48);
  return a1;
}
