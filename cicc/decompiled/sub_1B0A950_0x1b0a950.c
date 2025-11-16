// Function: sub_1B0A950
// Address: 0x1b0a950
//
__int64 __fastcall sub_1B0A950(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  char v8; // r8
  __int64 v9; // rax
  int v10; // ecx
  unsigned int v11; // esi
  __int64 v12; // rdx
  __int64 v13; // rdi
  __int64 v14; // rcx
  __int64 v15; // rsi
  __int64 result; // rax
  __int64 v17; // rdx
  __int64 v18; // rdx
  char v19; // r11
  unsigned int v20; // r10d
  __int64 v21; // rdi
  __int64 v22; // rdx
  __int64 v23; // r9
  __int64 v24; // rdi
  __int64 v25; // r15
  unsigned int v26; // edi
  __int64 v27; // rdx
  __int64 v28; // r8
  __int64 v29; // rdx
  int v30; // edx
  char v31; // di
  __int64 v32; // r8
  int v33; // esi
  unsigned int v34; // ecx
  __int64 v35; // rdx
  __int64 v36; // r9
  unsigned int v37; // esi
  unsigned int v38; // ecx
  int v39; // r8d
  unsigned int v40; // r9d
  unsigned int v41; // ecx
  int v42; // eax
  int v43; // eax
  int v44; // edx
  int v45; // r11d
  __int64 v46; // r10
  __int64 v47; // rdi
  int v48; // ecx
  unsigned int v49; // eax
  __int64 v50; // rsi
  int v51; // r9d
  int v52; // r10d
  __int64 v53; // r11
  __int64 v54; // rdi
  int v55; // ecx
  unsigned int v56; // eax
  __int64 v57; // rsi
  int v58; // r9d
  __int64 v59; // r8
  int v60; // ecx
  int v61; // ecx
  __int64 v62; // r10
  int v63; // edi
  unsigned int v64; // ecx
  __int64 v65; // rsi
  __int64 v66; // r10
  int v67; // edi
  unsigned int v68; // ecx
  __int64 v69; // rsi
  int v70; // r9d
  __int64 v71; // r8
  int v72; // edi
  int v73; // edi
  int v74; // r9d
  int v75; // r9d
  unsigned int v76; // [rsp+Ch] [rbp-34h]
  unsigned int v77; // [rsp+Ch] [rbp-34h]

  v8 = *(_BYTE *)(a4 + 8) & 1;
  if ( v8 )
  {
    v9 = a4 + 16;
    v10 = 3;
  }
  else
  {
    v17 = *(unsigned int *)(a4 + 24);
    v9 = *(_QWORD *)(a4 + 16);
    if ( !(_DWORD)v17 )
      goto LABEL_26;
    v10 = v17 - 1;
  }
  v11 = v10 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v12 = v9 + 16LL * v11;
  v13 = *(_QWORD *)v12;
  if ( *(_QWORD *)v12 == a1 )
    goto LABEL_4;
  v30 = 1;
  while ( v13 != -8 )
  {
    v51 = v30 + 1;
    v11 = v10 & (v30 + v11);
    v12 = v9 + 16LL * v11;
    v13 = *(_QWORD *)v12;
    if ( *(_QWORD *)v12 == a1 )
      goto LABEL_4;
    v30 = v51;
  }
  if ( v8 )
  {
    v29 = 64;
    goto LABEL_27;
  }
  v17 = *(unsigned int *)(a4 + 24);
LABEL_26:
  v29 = 16 * v17;
LABEL_27:
  v12 = v9 + v29;
LABEL_4:
  if ( v8 )
  {
    v14 = *(_QWORD *)a4;
    LODWORD(v15) = 4;
    if ( v12 != v9 + 64 )
      return *(unsigned int *)(v12 + 8);
  }
  else
  {
    v14 = *(_QWORD *)a4;
    v15 = *(unsigned int *)(a4 + 24);
    if ( v12 != v9 + 16 * v15 )
      return *(unsigned int *)(v12 + 8);
  }
  v18 = 0x17FFFFFFE8LL;
  v19 = *(_BYTE *)(a1 + 23) & 0x40;
  v20 = *(_DWORD *)(a1 + 20) & 0xFFFFFFF;
  if ( v20 )
  {
    v21 = 24LL * *(unsigned int *)(a1 + 56) + 8;
    v22 = 0;
    do
    {
      v23 = a1 - 24LL * v20;
      if ( v19 )
        v23 = *(_QWORD *)(a1 - 8);
      if ( a3 == *(_QWORD *)(v23 + v21) )
      {
        v18 = 24 * v22;
        goto LABEL_17;
      }
      ++v22;
      v21 += 8;
    }
    while ( v20 != (_DWORD)v22 );
    v18 = 0x17FFFFFFE8LL;
  }
LABEL_17:
  if ( v19 )
    v24 = *(_QWORD *)(a1 - 8);
  else
    v24 = a1 - 24LL * v20;
  v25 = *(_QWORD *)(v24 + v18);
  if ( v8 || *(_DWORD *)(a4 + 24) )
  {
    v26 = (v15 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v27 = v9 + 16LL * v26;
    v28 = *(_QWORD *)v27;
    if ( *(_QWORD *)v27 == a1 )
      goto LABEL_22;
    v45 = 1;
    v46 = 0;
    while ( v28 != -8 )
    {
      if ( !v46 && v28 == -16 )
        v46 = v27;
      v26 = (v15 - 1) & (v45 + v26);
      v27 = v9 + 16LL * v26;
      v28 = *(_QWORD *)v27;
      if ( *(_QWORD *)v27 == a1 )
        goto LABEL_22;
      ++v45;
    }
    if ( v46 )
      v27 = v46;
    *(_QWORD *)a4 = v14 + 1;
    v41 = *(_DWORD *)(a4 + 8);
    v42 = (v41 >> 1) + 1;
  }
  else
  {
    v27 = 0;
    *(_QWORD *)a4 = v14 + 1;
    v41 = *(_DWORD *)(a4 + 8);
    v42 = (v41 >> 1) + 1;
  }
  if ( 4 * v42 >= (unsigned int)(3 * v15) )
  {
    sub_1B0A580(a4, 2 * v15);
    if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
    {
      v47 = a4 + 16;
      v48 = 3;
    }
    else
    {
      v60 = *(_DWORD *)(a4 + 24);
      v47 = *(_QWORD *)(a4 + 16);
      if ( !v60 )
        goto LABEL_135;
      v48 = v60 - 1;
    }
    v49 = v48 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
    v27 = v47 + 16LL * v49;
    v50 = *(_QWORD *)v27;
    if ( *(_QWORD *)v27 != a1 )
    {
      v74 = 1;
      v59 = 0;
      while ( v50 != -8 )
      {
        if ( v50 == -16 && !v59 )
          v59 = v27;
        v49 = v48 & (v74 + v49);
        v27 = v47 + 16LL * v49;
        v50 = *(_QWORD *)v27;
        if ( *(_QWORD *)v27 == a1 )
          goto LABEL_66;
        ++v74;
      }
      goto LABEL_80;
    }
LABEL_66:
    v41 = *(_DWORD *)(a4 + 8);
    goto LABEL_51;
  }
  if ( (int)v15 - *(_DWORD *)(a4 + 12) - v42 <= (unsigned int)v15 >> 3 )
  {
    sub_1B0A580(a4, v15);
    if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
    {
      v54 = a4 + 16;
      v55 = 3;
      goto LABEL_77;
    }
    v61 = *(_DWORD *)(a4 + 24);
    v54 = *(_QWORD *)(a4 + 16);
    if ( v61 )
    {
      v55 = v61 - 1;
LABEL_77:
      v56 = v55 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v27 = v54 + 16LL * v56;
      v57 = *(_QWORD *)v27;
      if ( *(_QWORD *)v27 != a1 )
      {
        v58 = 1;
        v59 = 0;
        while ( v57 != -8 )
        {
          if ( v57 == -16 && !v59 )
            v59 = v27;
          v56 = v55 & (v58 + v56);
          v27 = v54 + 16LL * v56;
          v57 = *(_QWORD *)v27;
          if ( *(_QWORD *)v27 == a1 )
            goto LABEL_66;
          ++v58;
        }
LABEL_80:
        if ( v59 )
          v27 = v59;
        goto LABEL_66;
      }
      goto LABEL_66;
    }
LABEL_135:
    *(_DWORD *)(a4 + 8) = (2 * (*(_DWORD *)(a4 + 8) >> 1) + 2) | *(_DWORD *)(a4 + 8) & 1;
    BUG();
  }
LABEL_51:
  *(_DWORD *)(a4 + 8) = (2 * (v41 >> 1) + 2) | v41 & 1;
  if ( *(_QWORD *)v27 != -8 )
    --*(_DWORD *)(a4 + 12);
  *(_QWORD *)v27 = a1;
  *(_DWORD *)(v27 + 8) = 0;
LABEL_22:
  *(_DWORD *)(v27 + 8) = -1;
  if ( sub_13FC1A0(a2, v25) )
  {
    result = 1;
  }
  else
  {
    if ( *(_BYTE *)(v25 + 16) != 77 )
      return 0xFFFFFFFFLL;
    if ( **(_QWORD **)(a2 + 32) != *(_QWORD *)(v25 + 40) )
      return 0xFFFFFFFFLL;
    v43 = sub_1B0A950(v25, a2, a3, a4);
    v44 = v43;
    if ( v43 == -1 )
      return 0xFFFFFFFFLL;
    result = (unsigned int)(v43 + 1);
    if ( v44 == -2 )
      return 0xFFFFFFFFLL;
  }
  v31 = *(_BYTE *)(a4 + 8) & 1;
  if ( v31 )
  {
    v32 = a4 + 16;
    v33 = 3;
  }
  else
  {
    v37 = *(_DWORD *)(a4 + 24);
    v32 = *(_QWORD *)(a4 + 16);
    if ( !v37 )
    {
      v38 = *(_DWORD *)(a4 + 8);
      ++*(_QWORD *)a4;
      v35 = 0;
      v39 = (v38 >> 1) + 1;
      goto LABEL_42;
    }
    v33 = v37 - 1;
  }
  v34 = v33 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
  v35 = v32 + 16LL * v34;
  v36 = *(_QWORD *)v35;
  if ( *(_QWORD *)v35 != a1 )
  {
    v52 = 1;
    v53 = 0;
    while ( v36 != -8 )
    {
      if ( !v53 && v36 == -16 )
        v53 = v35;
      v34 = v33 & (v52 + v34);
      v35 = v32 + 16LL * v34;
      v36 = *(_QWORD *)v35;
      if ( *(_QWORD *)v35 == a1 )
        goto LABEL_38;
      ++v52;
    }
    v38 = *(_DWORD *)(a4 + 8);
    v40 = 12;
    v37 = 4;
    if ( v53 )
      v35 = v53;
    ++*(_QWORD *)a4;
    v39 = (v38 >> 1) + 1;
    if ( v31 )
    {
LABEL_43:
      if ( 4 * v39 < v40 )
      {
        if ( v37 - *(_DWORD *)(a4 + 12) - v39 > v37 >> 3 )
        {
LABEL_45:
          *(_DWORD *)(a4 + 8) = (2 * (v38 >> 1) + 2) | v38 & 1;
          if ( *(_QWORD *)v35 != -8 )
            --*(_DWORD *)(a4 + 12);
          *(_QWORD *)v35 = a1;
          *(_DWORD *)(v35 + 8) = 0;
          goto LABEL_38;
        }
        v77 = result;
        sub_1B0A580(a4, v37);
        result = v77;
        if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
        {
          v66 = a4 + 16;
          v67 = 3;
          goto LABEL_92;
        }
        v73 = *(_DWORD *)(a4 + 24);
        v66 = *(_QWORD *)(a4 + 16);
        if ( v73 )
        {
          v67 = v73 - 1;
LABEL_92:
          v68 = v67 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
          v35 = v66 + 16LL * v68;
          v69 = *(_QWORD *)v35;
          if ( *(_QWORD *)v35 != a1 )
          {
            v70 = 1;
            v71 = 0;
            while ( v69 != -8 )
            {
              if ( v69 == -16 && !v71 )
                v71 = v35;
              v68 = v67 & (v70 + v68);
              v35 = v66 + 16LL * v68;
              v69 = *(_QWORD *)v35;
              if ( *(_QWORD *)v35 == a1 )
                goto LABEL_89;
              ++v70;
            }
LABEL_95:
            if ( v71 )
              v35 = v71;
            goto LABEL_89;
          }
          goto LABEL_89;
        }
LABEL_136:
        *(_DWORD *)(a4 + 8) = (2 * (*(_DWORD *)(a4 + 8) >> 1) + 2) | *(_DWORD *)(a4 + 8) & 1;
        BUG();
      }
      v76 = result;
      sub_1B0A580(a4, 2 * v37);
      result = v76;
      if ( (*(_BYTE *)(a4 + 8) & 1) != 0 )
      {
        v62 = a4 + 16;
        v63 = 3;
      }
      else
      {
        v72 = *(_DWORD *)(a4 + 24);
        v62 = *(_QWORD *)(a4 + 16);
        if ( !v72 )
          goto LABEL_136;
        v63 = v72 - 1;
      }
      v64 = v63 & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v35 = v62 + 16LL * v64;
      v65 = *(_QWORD *)v35;
      if ( *(_QWORD *)v35 != a1 )
      {
        v75 = 1;
        v71 = 0;
        while ( v65 != -8 )
        {
          if ( !v71 && v65 == -16 )
            v71 = v35;
          v64 = v63 & (v75 + v64);
          v35 = v62 + 16LL * v64;
          v65 = *(_QWORD *)v35;
          if ( *(_QWORD *)v35 == a1 )
            goto LABEL_89;
          ++v75;
        }
        goto LABEL_95;
      }
LABEL_89:
      v38 = *(_DWORD *)(a4 + 8);
      goto LABEL_45;
    }
    v37 = *(_DWORD *)(a4 + 24);
LABEL_42:
    v40 = 3 * v37;
    goto LABEL_43;
  }
LABEL_38:
  *(_DWORD *)(v35 + 8) = result;
  return result;
}
