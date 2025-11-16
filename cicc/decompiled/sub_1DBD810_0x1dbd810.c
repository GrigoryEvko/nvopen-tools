// Function: sub_1DBD810
// Address: 0x1dbd810
//
__int64 __fastcall sub_1DBD810(_QWORD *a1, __int64 a2, int a3, int a4)
{
  __int64 v6; // r13
  __int64 v7; // rsi
  unsigned __int64 v8; // rdx
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  unsigned __int64 v11; // rax
  unsigned __int64 v12; // rcx
  unsigned __int64 v13; // rax
  __int16 v14; // di
  unsigned int v15; // r8d
  __int64 v16; // r10
  unsigned int v17; // r9d
  __int64 *v18; // rsi
  __int64 v19; // r11
  unsigned __int64 v20; // r10
  __int64 v21; // rsi
  __int64 v22; // rdi
  __int64 v23; // rcx
  int v24; // r15d
  __int64 v25; // rdi
  __int64 v26; // r8
  __int64 v27; // r10
  __int64 v28; // r9
  __int64 v29; // r11
  _WORD *v30; // r14
  __int16 *v31; // r11
  unsigned __int16 v32; // r9
  __int16 v33; // r14
  __int16 *v34; // r8
  __int64 j; // rdx
  int v37; // esi
  __int64 v38; // rax
  char v40; // dl
  unsigned __int16 v41; // dx
  __int64 v42; // rdi
  unsigned __int64 i; // rcx
  __int64 v44; // r8
  unsigned int v45; // edi
  unsigned int v46; // r9d
  __int64 *v47; // rdx
  __int64 v48; // rbx
  __int64 v49; // rdi
  __int64 v50; // rdx
  unsigned __int64 v51; // rdi
  unsigned int v52; // edx
  __int64 v53; // rdi
  int v54; // edx
  int v55; // r15d
  int v56; // r13d
  __int64 v58; // [rsp+18h] [rbp-38h]

  if ( a3 < 0 )
  {
    v27 = a2;
    v38 = *(_QWORD *)(*(_QWORD *)(a1[1] + 24LL) + 16LL * (a3 & 0x7FFFFFFF) + 8);
    if ( !v38 )
      return v27;
    while ( 1 )
    {
      if ( (*(_BYTE *)(v38 + 3) & 0x10) == 0 )
      {
        v40 = *(_BYTE *)(v38 + 4);
        if ( (v40 & 8) == 0 )
          break;
      }
      v38 = *(_QWORD *)(v38 + 32);
      if ( !v38 )
        return v27;
    }
LABEL_59:
    if ( (v40 & 1) != 0 )
      goto LABEL_70;
    v41 = (*(_DWORD *)v38 >> 8) & 0xFFF;
    if ( v41 )
    {
      if ( a4 && (*(_DWORD *)(*(_QWORD *)(a1[2] + 248LL) + 4LL * v41) & a4) == 0 )
        goto LABEL_70;
    }
    v42 = *(_QWORD *)(*a1 + 272LL);
    for ( i = *(_QWORD *)(v38 + 16); (*(_BYTE *)(i + 46) & 4) != 0; i = *(_QWORD *)i & 0xFFFFFFFFFFFFFFF8LL )
      ;
    v44 = *(_QWORD *)(v42 + 368);
    v45 = *(_DWORD *)(v42 + 384);
    if ( v45 )
    {
      v46 = (v45 - 1) & (((unsigned int)i >> 9) ^ ((unsigned int)i >> 4));
      v47 = (__int64 *)(v44 + 16LL * v46);
      v48 = *v47;
      if ( i == *v47 )
        goto LABEL_67;
      v54 = 1;
      while ( v48 != -8 )
      {
        v56 = v54 + 1;
        v46 = (v45 - 1) & (v54 + v46);
        v47 = (__int64 *)(v44 + 16LL * v46);
        v48 = *v47;
        if ( i == *v47 )
          goto LABEL_67;
        v54 = v56;
      }
    }
    v47 = (__int64 *)(v44 + 16LL * v45);
LABEL_67:
    v49 = v47[1];
    v50 = v49 >> 1;
    v51 = v49 & 0xFFFFFFFFFFFFFFF8LL;
    v52 = *(_DWORD *)(v51 + 24) | v50 & 3;
    if ( v52 > (*(_DWORD *)((v27 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(v27 >> 1) & 3) )
    {
      v53 = v51 | 4;
      if ( v52 < (*(_DWORD *)((a1[3] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)((__int64)a1[3] >> 1) & 3) )
        v27 = v53;
    }
LABEL_70:
    while ( 1 )
    {
      v38 = *(_QWORD *)(v38 + 32);
      if ( !v38 )
        return v27;
      if ( (*(_BYTE *)(v38 + 3) & 0x10) == 0 )
      {
        v40 = *(_BYTE *)(v38 + 4);
        if ( (v40 & 8) == 0 )
          goto LABEL_59;
      }
    }
  }
  v6 = *(_QWORD *)(*a1 + 272LL);
  v7 = sub_1DA9310(v6, a2);
  v8 = v7 + 24;
  v9 = a1[3] & 0xFFFFFFFFFFFFFFF8LL;
  while ( 1 )
  {
    v9 = *(_QWORD *)(v9 + 8);
    if ( v6 + 336 == v9 )
      break;
    if ( *(_QWORD *)(v9 + 16) )
      goto LABEL_6;
  }
  v9 = *(_QWORD *)(v6 + 336);
LABEL_6:
  v10 = v9 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v10 )
  {
    v11 = *(_QWORD *)(v10 + 16);
    if ( v11 )
    {
      if ( v7 == *(_QWORD *)(v11 + 24) )
        v8 = v11;
    }
  }
  v58 = *(_QWORD *)(v7 + 32);
  while ( v58 != v8 )
  {
    v12 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
    if ( !v12 )
      BUG();
    v8 = *(_QWORD *)v8 & 0xFFFFFFFFFFFFFFF8LL;
    if ( (*(_QWORD *)v12 & 4) != 0 )
    {
      if ( (unsigned __int16)(**(_WORD **)(v12 + 16) - 12) > 1u )
      {
        v13 = v12;
        v14 = *(_WORD *)(v12 + 46) & 4;
        if ( v14 )
        {
          do
            v12 = *(_QWORD *)v12 & 0xFFFFFFFFFFFFFFF8LL;
          while ( (*(_BYTE *)(v12 + 46) & 4) != 0 );
        }
LABEL_16:
        v15 = *(_DWORD *)(v6 + 384);
        v16 = *(_QWORD *)(v6 + 368);
        if ( v15 )
        {
          v17 = (v15 - 1) & (((unsigned int)v12 >> 9) ^ ((unsigned int)v12 >> 4));
          v18 = (__int64 *)(v16 + 16LL * v17);
          v19 = *v18;
          if ( *v18 == v12 )
          {
LABEL_18:
            v20 = v18[1] & 0xFFFFFFFFFFFFFFF8LL;
            if ( *(_DWORD *)((a2 & 0xFFFFFFFFFFFFFFF8LL) + 24) >= *(_DWORD *)(v20 + 24) )
              return a2;
            if ( v14 )
            {
              do
                v13 = *(_QWORD *)v13 & 0xFFFFFFFFFFFFFFF8LL;
              while ( (*(_BYTE *)(v13 + 46) & 4) != 0 );
            }
            v21 = *(_QWORD *)(v8 + 24) + 24LL;
            do
            {
              v22 = *(_QWORD *)(v13 + 32);
              v23 = v22 + 40LL * *(unsigned int *)(v13 + 40);
              if ( v22 != v23 )
                break;
              v13 = *(_QWORD *)(v13 + 8);
              if ( v21 == v13 )
                break;
            }
            while ( (*(_BYTE *)(v13 + 46) & 4) != 0 );
            if ( v23 != v22 )
            {
              do
              {
                while ( 1 )
                {
                  if ( !*(_BYTE *)v22 && (*(_BYTE *)(v22 + 4) & 1) == 0 )
                  {
                    v24 = *(_DWORD *)(v22 + 8);
                    if ( v24 > 0 )
                    {
                      v28 = a1[2];
                      v29 = *(_QWORD *)(v28 + 8);
                      v30 = (_WORD *)(*(_QWORD *)(v28 + 56)
                                    + 2LL * (*(_DWORD *)(v29 + 24LL * (unsigned int)v24 + 16) >> 4));
                      LODWORD(v28) = *(_DWORD *)(v29 + 24LL * (unsigned int)v24 + 16) & 0xF;
                      v31 = v30 + 1;
                      v32 = *v30 + v24 * v28;
LABEL_40:
                      v34 = v31;
                      while ( v34 )
                      {
                        if ( a3 == v32 )
                          return v20 | 4;
                        v33 = *v34;
                        v31 = 0;
                        ++v34;
                        v32 += v33;
                        if ( !v33 )
                          goto LABEL_40;
                      }
                    }
                  }
                  v25 = v22 + 40;
                  v26 = v23;
                  if ( v25 == v23 )
                    break;
                  v23 = v25;
LABEL_37:
                  v22 = v23;
                  v23 = v26;
                }
                while ( 1 )
                {
                  v13 = *(_QWORD *)(v13 + 8);
                  if ( v21 == v13 || (*(_BYTE *)(v13 + 46) & 4) == 0 )
                    break;
                  v23 = *(_QWORD *)(v13 + 32);
                  v26 = v23 + 40LL * *(unsigned int *)(v13 + 40);
                  if ( v23 != v26 )
                    goto LABEL_37;
                }
                v22 = v23;
                v23 = v26;
              }
              while ( v26 != v22 );
            }
            continue;
          }
          v37 = 1;
          while ( v19 != -8 )
          {
            v55 = v37 + 1;
            v17 = (v15 - 1) & (v37 + v17);
            v18 = (__int64 *)(v16 + 16LL * v17);
            v19 = *v18;
            if ( *v18 == v12 )
              goto LABEL_18;
            v37 = v55;
          }
        }
        v18 = (__int64 *)(v16 + 16LL * v15);
        goto LABEL_18;
      }
    }
    else
    {
      if ( (*(_BYTE *)(v12 + 46) & 4) != 0 )
      {
        for ( j = *(_QWORD *)v12; ; j = *(_QWORD *)v8 )
        {
          v8 = j & 0xFFFFFFFFFFFFFFF8LL;
          if ( (*(_BYTE *)(v8 + 46) & 4) == 0 )
            break;
        }
      }
      if ( (unsigned __int16)(**(_WORD **)(v8 + 16) - 12) > 1u )
      {
        v12 = v8;
        v13 = v8;
        v14 = *(_WORD *)(v8 + 46) & 4;
        goto LABEL_16;
      }
    }
  }
  return a2;
}
