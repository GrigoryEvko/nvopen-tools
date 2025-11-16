// Function: sub_2E14320
// Address: 0x2e14320
//
__int64 __fastcall sub_2E14320(_QWORD *a1, __int64 a2, int a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdi
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned __int64 v11; // rcx
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rax
  __int64 v14; // rbx
  __int16 v15; // ax
  unsigned __int64 v16; // rdx
  __int64 i; // rcx
  __int16 v18; // ax
  unsigned __int64 v19; // rax
  int v20; // r10d
  int v21; // edi
  unsigned __int64 v22; // r8
  __int64 j; // r11
  __int16 v24; // r8
  unsigned int v25; // r10d
  __int64 v26; // r11
  unsigned int v27; // r14d
  __int64 *v28; // r8
  __int64 v29; // r12
  unsigned __int64 v30; // r13
  __int64 v31; // rdi
  __int64 v32; // r8
  __int64 v33; // rdx
  __int64 v34; // r10
  __int64 v35; // r8
  __int64 v36; // r10
  __int64 v37; // r12
  unsigned int v38; // r10d
  int v39; // r11d
  __int16 *v40; // r10
  int v41; // r14d
  int v43; // r8d
  __int64 v44; // rax
  char v46; // dl
  __int64 v47; // rbx
  unsigned __int16 v48; // dx
  unsigned __int64 v49; // rsi
  __int64 v50; // r12
  unsigned __int64 k; // rdx
  __int64 m; // r11
  __int16 v53; // cx
  unsigned int v54; // r11d
  __int64 v55; // r13
  unsigned int v56; // esi
  __int64 *v57; // rcx
  __int64 v58; // r15
  unsigned __int64 v59; // rsi
  unsigned int v60; // edx
  __int64 v61; // rsi
  _QWORD *v62; // rdx
  int v63; // ecx
  __int64 v64; // rdx
  _QWORD *v65; // r8
  __int64 *v66; // rcx
  int v67; // r14d
  int v68; // [rsp+0h] [rbp-3Ch]
  unsigned __int64 v70; // [rsp+Ch] [rbp-30h]

  v6 = a2;
  if ( a3 >= 0 )
  {
    v8 = *(_QWORD *)(*a1 + 32LL);
    v70 = a2 & 0xFFFFFFFFFFFFFFF8LL;
    v9 = *(_QWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 16);
    if ( v9 )
    {
      v10 = *(_QWORD *)(v9 + 24);
    }
    else
    {
      v64 = *(unsigned int *)(v8 + 304);
      v65 = *(_QWORD **)(v8 + 296);
      if ( *(_DWORD *)(v8 + 304) )
      {
        do
        {
          v66 = &v65[2 * (v64 >> 1)];
          if ( (*(_DWORD *)(v70 + 24) | (unsigned int)(a2 >> 1) & 3) >= (*(_DWORD *)((*v66 & 0xFFFFFFFFFFFFFFF8LL) + 24)
                                                                       | (unsigned int)(*v66 >> 1) & 3) )
          {
            v65 = v66 + 2;
            v64 = v64 - (v64 >> 1) - 1;
          }
          else
          {
            v64 >>= 1;
          }
        }
        while ( v64 > 0 );
      }
      v10 = *(v65 - 1);
    }
    v11 = v10 + 48;
    v12 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
    while ( 1 )
    {
      v12 = *(_QWORD *)(v12 + 8);
      if ( v8 + 96 == v12 )
        break;
      if ( *(_QWORD *)(v12 + 16) )
        goto LABEL_8;
    }
    v12 = *(_QWORD *)(v8 + 96);
LABEL_8:
    v13 = *(_QWORD *)((v12 & 0xFFFFFFFFFFFFFFF8LL) + 16);
    if ( v13 && *(_QWORD *)(v13 + 24) == v10 )
      v11 = v13;
    v14 = *(_QWORD *)(v10 + 56);
    if ( v14 == v11 )
      return a2;
    while ( 1 )
    {
      v16 = *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL;
      if ( !v16 )
        BUG();
      v11 = *(_QWORD *)v11 & 0xFFFFFFFFFFFFFFF8LL;
      if ( (*(_QWORD *)v16 & 4) != 0 )
      {
        v15 = *(_WORD *)(v16 + 68);
        if ( (unsigned __int16)(v15 - 14) <= 4u || v15 == 24 )
          goto LABEL_15;
        v19 = v16;
        v20 = *(_DWORD *)(v16 + 44) & 0xFFFFFF;
        v21 = *(_DWORD *)(v16 + 44) & 4;
        if ( v21 )
        {
          do
            v16 = *(_QWORD *)v16 & 0xFFFFFFFFFFFFFFF8LL;
          while ( (*(_BYTE *)(v16 + 44) & 4) != 0 );
        }
      }
      else
      {
        if ( (*(_BYTE *)(v16 + 44) & 4) != 0 )
        {
          for ( i = *(_QWORD *)v16; ; i = *(_QWORD *)v11 )
          {
            v11 = i & 0xFFFFFFFFFFFFFFF8LL;
            if ( (*(_BYTE *)(v11 + 44) & 4) == 0 )
              break;
          }
        }
        v18 = *(_WORD *)(v11 + 68);
        if ( (unsigned __int16)(v18 - 14) <= 4u || v18 == 24 )
          goto LABEL_15;
        v16 = v11;
        v19 = v11;
        v20 = *(_DWORD *)(v11 + 44);
        v21 = v20 & 4;
      }
      v22 = v19;
      if ( (v20 & 8) != 0 )
      {
        do
          v22 = *(_QWORD *)(v22 + 8);
        while ( (*(_BYTE *)(v22 + 44) & 8) != 0 );
      }
      for ( j = *(_QWORD *)(v22 + 8); j != v16; v16 = *(_QWORD *)(v16 + 8) )
      {
        v24 = *(_WORD *)(v16 + 68);
        if ( (unsigned __int16)(v24 - 14) > 4u && v24 != 24 )
          break;
      }
      v25 = *(_DWORD *)(v8 + 144);
      v26 = *(_QWORD *)(v8 + 128);
      if ( !v25 )
        goto LABEL_63;
      v27 = (v25 - 1) & (((unsigned int)v16 >> 9) ^ ((unsigned int)v16 >> 4));
      v28 = (__int64 *)(v26 + 16LL * v27);
      v29 = *v28;
      if ( v16 != *v28 )
        break;
LABEL_33:
      v30 = v28[1] & 0xFFFFFFFFFFFFFFF8LL;
      if ( *(_DWORD *)(v70 + 24) >= *(_DWORD *)(v30 + 24) )
        return a2;
      if ( v21 )
      {
        do
          v19 = *(_QWORD *)v19 & 0xFFFFFFFFFFFFFFF8LL;
        while ( (*(_BYTE *)(v19 + 44) & 4) != 0 );
      }
      v31 = *(_QWORD *)(v11 + 24) + 48LL;
      while ( 1 )
      {
        v32 = *(_QWORD *)(v19 + 32);
        v33 = v32 + 40LL * (*(_DWORD *)(v19 + 40) & 0xFFFFFF);
        if ( v32 != v33 )
          break;
        v19 = *(_QWORD *)(v19 + 8);
        if ( v31 == v19 )
          break;
        if ( (*(_BYTE *)(v19 + 44) & 4) == 0 )
        {
          v19 = *(_QWORD *)(v11 + 24) + 48LL;
          break;
        }
      }
LABEL_40:
      while ( v33 != v32 )
      {
        while ( 1 )
        {
          if ( !*(_BYTE *)v32 && (*(_BYTE *)(v32 + 4) & 1) == 0 )
          {
            v34 = *(unsigned int *)(v32 + 8);
            if ( (unsigned int)(v34 - 1) <= 0x3FFFFFFE )
            {
              v37 = a1[2];
              v38 = *(_DWORD *)(*(_QWORD *)(v37 + 8) + 24 * v34 + 16);
              v39 = v38 & 0xFFF;
              v40 = (__int16 *)(*(_QWORD *)(v37 + 56) + 2LL * (v38 >> 12));
              while ( v40 )
              {
                if ( a3 == v39 )
                  return v30 | 4;
                v41 = *v40++;
                v39 += v41;
                if ( !(_WORD)v41 )
                  break;
              }
            }
          }
          v35 = v32 + 40;
          v36 = v33;
          if ( v35 == v33 )
            break;
          v33 = v35;
LABEL_50:
          v32 = v33;
          v33 = v36;
        }
        while ( 1 )
        {
          v19 = *(_QWORD *)(v19 + 8);
          if ( v31 == v19 )
          {
            v32 = v33;
            v33 = v36;
            goto LABEL_40;
          }
          if ( (*(_BYTE *)(v19 + 44) & 4) == 0 )
            break;
          v33 = *(_QWORD *)(v19 + 32);
          v36 = v33 + 40LL * (*(_DWORD *)(v19 + 40) & 0xFFFFFF);
          if ( v33 != v36 )
            goto LABEL_50;
        }
        v32 = v33;
        v19 = *(_QWORD *)(v11 + 24) + 48LL;
        v33 = v36;
      }
LABEL_15:
      if ( v14 == v11 )
        return a2;
    }
    v43 = 1;
    while ( v29 != -4096 )
    {
      v27 = (v25 - 1) & (v43 + v27);
      v68 = v43 + 1;
      v28 = (__int64 *)(v26 + 16LL * v27);
      v29 = *v28;
      if ( v16 == *v28 )
        goto LABEL_33;
      v43 = v68;
    }
LABEL_63:
    v28 = (__int64 *)(v26 + 16LL * v25);
    goto LABEL_33;
  }
  v44 = *(_QWORD *)(*(_QWORD *)(a1[1] + 56LL) + 16LL * (a3 & 0x7FFFFFFF) + 8);
  if ( v44 )
  {
    while ( 1 )
    {
      if ( (*(_BYTE *)(v44 + 3) & 0x10) == 0 )
      {
        v46 = *(_BYTE *)(v44 + 4);
        if ( (v46 & 8) == 0 )
          break;
      }
      v44 = *(_QWORD *)(v44 + 32);
      if ( !v44 )
        return v6;
    }
    v47 = a4 | a5;
LABEL_71:
    if ( (v46 & 1) != 0 )
      goto LABEL_86;
    v48 = (*(_DWORD *)v44 >> 8) & 0xFFF;
    if ( v48 )
    {
      if ( v47 )
      {
        v62 = (_QWORD *)(*(_QWORD *)(a1[2] + 272LL) + 16LL * v48);
        if ( !(a5 & v62[1] | a4 & *v62) )
          goto LABEL_86;
      }
    }
    v49 = *(_QWORD *)(v44 + 16);
    v50 = *(_QWORD *)(*a1 + 32LL);
    for ( k = v49; (*(_BYTE *)(k + 44) & 4) != 0; k = *(_QWORD *)k & 0xFFFFFFFFFFFFFFF8LL )
      ;
    for ( ; (*(_BYTE *)(v49 + 44) & 8) != 0; v49 = *(_QWORD *)(v49 + 8) )
      ;
    for ( m = *(_QWORD *)(v49 + 8); m != k; k = *(_QWORD *)(k + 8) )
    {
      v53 = *(_WORD *)(k + 68);
      if ( (unsigned __int16)(v53 - 14) > 4u && v53 != 24 )
        break;
    }
    v54 = *(_DWORD *)(v50 + 144);
    v55 = *(_QWORD *)(v50 + 128);
    if ( v54 )
    {
      v56 = (v54 - 1) & (((unsigned int)k >> 9) ^ ((unsigned int)k >> 4));
      v57 = (__int64 *)(v55 + 16LL * v56);
      v58 = *v57;
      if ( *v57 == k )
      {
LABEL_83:
        v59 = v57[1] & 0xFFFFFFFFFFFFFFF8LL;
        v60 = *(_DWORD *)(v59 + 24) | (v57[1] >> 1) & 3;
        if ( v60 > (*(_DWORD *)((v6 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v6 >> 1) & 3) )
        {
          v61 = v59 | 4;
          if ( v60 < (*(_DWORD *)((a1[3] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)a1[3] >> 1) & 3) )
            v6 = v61;
        }
LABEL_86:
        while ( 1 )
        {
          v44 = *(_QWORD *)(v44 + 32);
          if ( !v44 )
            return v6;
          while ( (*(_BYTE *)(v44 + 3) & 0x10) == 0 )
          {
            v46 = *(_BYTE *)(v44 + 4);
            if ( (v46 & 8) == 0 )
              goto LABEL_71;
            v44 = *(_QWORD *)(v44 + 32);
            if ( !v44 )
              return v6;
          }
        }
      }
      v63 = 1;
      while ( v58 != -4096 )
      {
        v67 = v63 + 1;
        v56 = (v54 - 1) & (v63 + v56);
        v57 = (__int64 *)(v55 + 16LL * v56);
        v58 = *v57;
        if ( *v57 == k )
          goto LABEL_83;
        v63 = v67;
      }
    }
    v57 = (__int64 *)(v55 + 16LL * v54);
    goto LABEL_83;
  }
  return v6;
}
