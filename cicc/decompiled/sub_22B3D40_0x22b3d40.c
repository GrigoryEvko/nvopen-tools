// Function: sub_22B3D40
// Address: 0x22b3d40
//
__int64 __fastcall sub_22B3D40(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  int v4; // r10d
  __int64 v5; // r12
  int v6; // r11d
  int v7; // r13d
  __int64 v8; // r10
  unsigned int v9; // edi
  _QWORD *v10; // rax
  __int64 v11; // rsi
  unsigned int v12; // r8d
  __int64 v13; // rbx
  __int64 v14; // r15
  unsigned __int64 v15; // rdx
  unsigned __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // r8
  __int64 v19; // rdx
  __int64 v20; // r8
  _QWORD *i; // rdx
  __int64 v22; // rax
  __int64 v23; // rbx
  __int64 v24; // rdx
  int v25; // edi
  int v26; // edi
  __int64 v27; // r10
  unsigned int v28; // esi
  __int64 *v29; // rcx
  __int64 v30; // r9
  int v31; // edx
  int v32; // esi
  int v33; // edx
  unsigned int v34; // edi
  _QWORD *v35; // rcx
  __int64 v36; // r8
  int v37; // eax
  int v38; // eax
  int v39; // eax
  __int64 v40; // r8
  _QWORD *v41; // r9
  int v42; // r15d
  unsigned int v43; // edx
  __int64 v44; // rdi
  __int64 v45; // rsi
  _QWORD *v46; // rsi
  _QWORD *v47; // rcx
  int v48; // r15d
  int v49; // [rsp+Ch] [rbp-64h]
  __int64 v50; // [rsp+10h] [rbp-60h]
  int v51; // [rsp+18h] [rbp-58h]
  int v52; // [rsp+18h] [rbp-58h]
  int v53; // [rsp+18h] [rbp-58h]
  __int64 *v54; // [rsp+18h] [rbp-58h]
  __int64 v55; // [rsp+20h] [rbp-50h]
  __int64 v56; // [rsp+20h] [rbp-50h]
  __int64 v57; // [rsp+20h] [rbp-50h]
  __int64 v58; // [rsp+20h] [rbp-50h]
  unsigned int v59; // [rsp+28h] [rbp-48h]
  __int64 v60; // [rsp+28h] [rbp-48h]
  int v61; // [rsp+28h] [rbp-48h]
  unsigned int v62; // [rsp+28h] [rbp-48h]
  __int64 v63; // [rsp+30h] [rbp-40h]
  __int64 v64; // [rsp+38h] [rbp-38h]

  result = *(_QWORD *)(a2 + 32);
  v63 = a2 + 24;
  v50 = a1 + 40;
  v64 = result;
  if ( result != a2 + 24 )
  {
    v4 = 0;
    while ( 1 )
    {
      if ( !v64 )
        BUG();
      v5 = *(_QWORD *)(v64 + 24);
      if ( v5 != v64 + 16 )
        break;
LABEL_36:
      result = *(_QWORD *)(v64 + 8);
      v64 = result;
      if ( v63 == result )
        return result;
    }
    v6 = v4;
    v7 = v4;
    v8 = v64 + 16;
    while ( 1 )
    {
      v12 = *(_DWORD *)(a1 + 64);
      v13 = v5 - 24;
      v14 = *(_QWORD *)(a1 + 48);
      if ( !v5 )
        v13 = 0;
      ++v7;
      if ( !v12 )
        break;
      v9 = (v12 - 1) & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
      v10 = (_QWORD *)(v14 + 16LL * v9);
      v11 = *v10;
      if ( v13 == *v10 )
      {
LABEL_7:
        v5 = *(_QWORD *)(v5 + 8);
        if ( v8 == v5 )
          goto LABEL_35;
        goto LABEL_8;
      }
      v61 = 1;
      v35 = 0;
      while ( v11 != -4096 )
      {
        if ( v35 || v11 != -8192 )
          v10 = v35;
        v9 = (v12 - 1) & (v61 + v9);
        v11 = *(_QWORD *)(v14 + 16LL * v9);
        if ( v13 == v11 )
          goto LABEL_7;
        ++v61;
        v35 = v10;
        v10 = (_QWORD *)(v14 + 16LL * v9);
      }
      if ( !v35 )
        v35 = v10;
      v37 = *(_DWORD *)(a1 + 56);
      ++*(_QWORD *)(a1 + 40);
      v32 = v37 + 1;
      if ( 4 * (v37 + 1) >= 3 * v12 )
        goto LABEL_13;
      if ( v12 - *(_DWORD *)(a1 + 60) - v32 <= v12 >> 3 )
      {
        v53 = v6;
        v58 = v8;
        v62 = ((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4);
        sub_B23080(v50, v12);
        v38 = *(_DWORD *)(a1 + 64);
        if ( !v38 )
          goto LABEL_84;
        v39 = v38 - 1;
        v40 = *(_QWORD *)(a1 + 48);
        v41 = 0;
        v8 = v58;
        v42 = 1;
        v43 = v39 & v62;
        v6 = v53;
        v32 = *(_DWORD *)(a1 + 56) + 1;
        v35 = (_QWORD *)(v40 + 16LL * (v39 & v62));
        v44 = *v35;
        if ( v13 != *v35 )
        {
          while ( v44 != -4096 )
          {
            if ( !v41 && v44 == -8192 )
              v41 = v35;
            v43 = v39 & (v42 + v43);
            v35 = (_QWORD *)(v40 + 16LL * v43);
            v44 = *v35;
            if ( v13 == *v35 )
              goto LABEL_32;
            ++v42;
          }
          goto LABEL_47;
        }
      }
LABEL_32:
      *(_DWORD *)(a1 + 56) = v32;
      if ( *v35 != -4096 )
        --*(_DWORD *)(a1 + 60);
      *v35 = v13;
      *((_DWORD *)v35 + 2) = v6;
      v5 = *(_QWORD *)(v5 + 8);
      if ( v8 == v5 )
      {
LABEL_35:
        v4 = v7;
        goto LABEL_36;
      }
LABEL_8:
      v6 = v7;
    }
    ++*(_QWORD *)(a1 + 40);
LABEL_13:
    v51 = v6;
    v55 = v8;
    v59 = v12;
    v15 = ((((((((2 * v12 - 1) | ((unsigned __int64)(2 * v12 - 1) >> 1)) >> 2)
             | (2 * v12 - 1)
             | ((unsigned __int64)(2 * v12 - 1) >> 1)) >> 4)
           | (((2 * v12 - 1) | ((unsigned __int64)(2 * v12 - 1) >> 1)) >> 2)
           | (2 * v12 - 1)
           | ((unsigned __int64)(2 * v12 - 1) >> 1)) >> 8)
         | (((((2 * v12 - 1) | ((unsigned __int64)(2 * v12 - 1) >> 1)) >> 2)
           | (2 * v12 - 1)
           | ((unsigned __int64)(2 * v12 - 1) >> 1)) >> 4)
         | (((2 * v12 - 1) | ((unsigned __int64)(2 * v12 - 1) >> 1)) >> 2)
         | (2 * v12 - 1)
         | ((unsigned __int64)(2 * v12 - 1) >> 1)) >> 16;
    v16 = (v15
         | (((((((2 * v12 - 1) | ((unsigned __int64)(2 * v12 - 1) >> 1)) >> 2)
             | (2 * v12 - 1)
             | ((unsigned __int64)(2 * v12 - 1) >> 1)) >> 4)
           | (((2 * v12 - 1) | ((unsigned __int64)(2 * v12 - 1) >> 1)) >> 2)
           | (2 * v12 - 1)
           | ((unsigned __int64)(2 * v12 - 1) >> 1)) >> 8)
         | (((((2 * v12 - 1) | ((unsigned __int64)(2 * v12 - 1) >> 1)) >> 2)
           | (2 * v12 - 1)
           | ((unsigned __int64)(2 * v12 - 1) >> 1)) >> 4)
         | (((2 * v12 - 1) | ((unsigned __int64)(2 * v12 - 1) >> 1)) >> 2)
         | (2 * v12 - 1)
         | ((unsigned __int64)(2 * v12 - 1) >> 1))
        + 1;
    if ( (unsigned int)v16 < 0x40 )
      LODWORD(v16) = 64;
    *(_DWORD *)(a1 + 64) = v16;
    v17 = (_QWORD *)sub_C7D670(16LL * (unsigned int)v16, 8);
    v18 = v59;
    v8 = v55;
    *(_QWORD *)(a1 + 48) = v17;
    v6 = v51;
    if ( v14 )
    {
      v19 = *(unsigned int *)(a1 + 64);
      *(_QWORD *)(a1 + 56) = 0;
      v60 = 16LL * v59;
      v20 = v14 + 16 * v18;
      for ( i = &v17[2 * v19]; i != v17; v17 += 2 )
      {
        if ( v17 )
          *v17 = -4096;
      }
      v22 = v14;
      if ( v14 != v20 )
      {
        v56 = v13;
        v23 = v8;
        do
        {
          v24 = *(_QWORD *)v22;
          if ( *(_QWORD *)v22 != -8192 && v24 != -4096 )
          {
            v25 = *(_DWORD *)(a1 + 64);
            if ( !v25 )
            {
              MEMORY[0] = *(_QWORD *)v22;
              BUG();
            }
            v26 = v25 - 1;
            v27 = *(_QWORD *)(a1 + 48);
            v28 = v26 & (((unsigned int)v24 >> 9) ^ ((unsigned int)v24 >> 4));
            v29 = (__int64 *)(v27 + 16LL * v28);
            v30 = *v29;
            if ( v24 != *v29 )
            {
              v49 = 1;
              v54 = 0;
              while ( v30 != -4096 )
              {
                if ( !v54 )
                {
                  if ( v30 != -8192 )
                    v29 = 0;
                  v54 = v29;
                }
                v28 = v26 & (v49 + v28);
                v29 = (__int64 *)(v27 + 16LL * v28);
                v30 = *v29;
                if ( v24 == *v29 )
                  goto LABEL_26;
                ++v49;
              }
              if ( v54 )
                v29 = v54;
            }
LABEL_26:
            *v29 = v24;
            *((_DWORD *)v29 + 2) = *(_DWORD *)(v22 + 8);
            ++*(_DWORD *)(a1 + 56);
          }
          v22 += 16;
        }
        while ( v20 != v22 );
        v8 = v23;
        v13 = v56;
      }
      v52 = v6;
      v57 = v8;
      sub_C7D6A0(v14, v60, 8);
      v17 = *(_QWORD **)(a1 + 48);
      v31 = *(_DWORD *)(a1 + 64);
      v8 = v57;
      v6 = v52;
      v32 = *(_DWORD *)(a1 + 56) + 1;
    }
    else
    {
      v45 = *(unsigned int *)(a1 + 64);
      *(_QWORD *)(a1 + 56) = 0;
      v31 = v45;
      v46 = &v17[2 * v45];
      if ( v17 != v46 )
      {
        v47 = v17;
        do
        {
          if ( v47 )
            *v47 = -4096;
          v47 += 2;
        }
        while ( v46 != v47 );
      }
      v32 = 1;
    }
    if ( !v31 )
    {
LABEL_84:
      ++*(_DWORD *)(a1 + 56);
      BUG();
    }
    v33 = v31 - 1;
    v34 = v33 & (((unsigned int)v13 >> 9) ^ ((unsigned int)v13 >> 4));
    v35 = &v17[2 * v34];
    v36 = *v35;
    if ( v13 != *v35 )
    {
      v48 = 1;
      v41 = 0;
      while ( v36 != -4096 )
      {
        if ( !v41 && v36 == -8192 )
          v41 = v35;
        v34 = v33 & (v48 + v34);
        v35 = &v17[2 * v34];
        v36 = *v35;
        if ( v13 == *v35 )
          goto LABEL_32;
        ++v48;
      }
LABEL_47:
      if ( v41 )
        v35 = v41;
      goto LABEL_32;
    }
    goto LABEL_32;
  }
  return result;
}
