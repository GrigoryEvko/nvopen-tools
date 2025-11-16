// Function: sub_A4A530
// Address: 0xa4a530
//
__int64 __fastcall sub_A4A530(__int64 a1, __int64 a2)
{
  unsigned int v3; // esi
  __int64 v4; // r8
  __int64 *v5; // rdx
  int v6; // r10d
  unsigned int v7; // edi
  __int64 *v8; // rax
  __int64 v9; // rcx
  int v10; // eax
  int v12; // eax
  int v13; // ecx
  int v14; // r14d
  __int64 v15; // rax
  __int64 v16; // r12
  __int64 v17; // r11
  __int64 v18; // r8
  unsigned int v19; // edi
  _QWORD *v20; // rax
  __int64 v21; // rcx
  _DWORD *v22; // rax
  __int64 v23; // rbx
  int v24; // esi
  int v25; // esi
  __int64 v26; // r8
  unsigned int v27; // ecx
  int v28; // eax
  _QWORD *v29; // rdx
  __int64 v30; // rdi
  int v31; // eax
  int v32; // ecx
  int v33; // ecx
  _QWORD *v34; // r8
  int v35; // r9d
  unsigned int v36; // r15d
  __int64 v37; // rdi
  __int64 v38; // rsi
  int v39; // eax
  int v40; // eax
  __int64 v41; // r8
  unsigned int v42; // edi
  __int64 v43; // rsi
  int v44; // r10d
  __int64 *v45; // r9
  int v46; // eax
  int v47; // eax
  __int64 v48; // r8
  int v49; // r10d
  unsigned int v50; // edi
  __int64 v51; // rsi
  int v52; // r15d
  _QWORD *v53; // r9
  __int64 v54; // [rsp+10h] [rbp-50h]
  int v55; // [rsp+10h] [rbp-50h]
  __int64 v56; // [rsp+10h] [rbp-50h]
  __int64 v57; // [rsp+18h] [rbp-48h]
  unsigned int v58; // [rsp+24h] [rbp-3Ch]

  v58 = ((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4);
  v57 = a1 + 440;
  v3 = *(_DWORD *)(a1 + 464);
  while ( 1 )
  {
LABEL_2:
    if ( !v3 )
    {
      ++*(_QWORD *)(a1 + 440);
      goto LABEL_51;
    }
    v4 = *(_QWORD *)(a1 + 448);
    v5 = 0;
    v6 = 1;
    v7 = (v3 - 1) & v58;
    v8 = (__int64 *)(v4 + 16LL * v7);
    v9 = *v8;
    if ( a2 != *v8 )
    {
      while ( v9 != -4096 )
      {
        if ( !v5 && v9 == -8192 )
          v5 = v8;
        v7 = (v3 - 1) & (v6 + v7);
        v8 = (__int64 *)(v4 + 16LL * v7);
        v9 = *v8;
        if ( *v8 == a2 )
          goto LABEL_4;
        ++v6;
      }
      if ( !v5 )
        v5 = v8;
      v12 = *(_DWORD *)(a1 + 456);
      ++*(_QWORD *)(a1 + 440);
      v13 = v12 + 1;
      if ( 4 * (v12 + 1) < 3 * v3 )
      {
        if ( v3 - *(_DWORD *)(a1 + 460) - v13 > v3 >> 3 )
          goto LABEL_16;
        sub_A4A350(v57, v3);
        v46 = *(_DWORD *)(a1 + 464);
        if ( !v46 )
        {
LABEL_86:
          ++*(_DWORD *)(a1 + 456);
          BUG();
        }
        v47 = v46 - 1;
        v48 = *(_QWORD *)(a1 + 448);
        v45 = 0;
        v49 = 1;
        v50 = v47 & v58;
        v13 = *(_DWORD *)(a1 + 456) + 1;
        v5 = (__int64 *)(v48 + 16LL * (v47 & v58));
        v51 = *v5;
        if ( *v5 == a2 )
          goto LABEL_16;
        while ( v51 != -4096 )
        {
          if ( !v45 && v51 == -8192 )
            v45 = v5;
          v50 = v47 & (v49 + v50);
          v5 = (__int64 *)(v48 + 16LL * v50);
          v51 = *v5;
          if ( *v5 == a2 )
            goto LABEL_16;
          ++v49;
        }
        goto LABEL_55;
      }
LABEL_51:
      sub_A4A350(v57, 2 * v3);
      v39 = *(_DWORD *)(a1 + 464);
      if ( !v39 )
        goto LABEL_86;
      v40 = v39 - 1;
      v41 = *(_QWORD *)(a1 + 448);
      v42 = v40 & v58;
      v13 = *(_DWORD *)(a1 + 456) + 1;
      v5 = (__int64 *)(v41 + 16LL * (v40 & v58));
      v43 = *v5;
      if ( *v5 == a2 )
        goto LABEL_16;
      v44 = 1;
      v45 = 0;
      while ( v43 != -4096 )
      {
        if ( !v45 && v43 == -8192 )
          v45 = v5;
        v42 = v40 & (v44 + v42);
        v5 = (__int64 *)(v41 + 16LL * v42);
        v43 = *v5;
        if ( *v5 == a2 )
          goto LABEL_16;
        ++v44;
      }
LABEL_55:
      if ( v45 )
        v5 = v45;
LABEL_16:
      *(_DWORD *)(a1 + 456) = v13;
      if ( *v5 != -4096 )
        --*(_DWORD *)(a1 + 460);
      *((_DWORD *)v5 + 2) = 0;
      *v5 = a2;
      v3 = *(_DWORD *)(a1 + 464);
      goto LABEL_19;
    }
LABEL_4:
    v10 = *((_DWORD *)v8 + 2);
    if ( v10 )
      return (unsigned int)(v10 - 1);
LABEL_19:
    v14 = 0;
    v15 = *(_QWORD *)(a2 + 72);
    v16 = *(_QWORD *)(v15 + 80);
    v17 = v15 + 72;
    if ( v16 != v15 + 72 )
    {
      while ( 1 )
      {
        v23 = v16 - 24;
        if ( !v16 )
          v23 = 0;
        ++v14;
        if ( !v3 )
          break;
        v18 = *(_QWORD *)(a1 + 448);
        v19 = (v3 - 1) & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
        v20 = (_QWORD *)(v18 + 16LL * v19);
        v21 = *v20;
        if ( v23 != *v20 )
        {
          v55 = 1;
          v29 = 0;
          while ( v21 != -4096 )
          {
            if ( !v29 && v21 == -8192 )
              v29 = v20;
            v19 = (v3 - 1) & (v55 + v19);
            v20 = (_QWORD *)(v18 + 16LL * v19);
            v21 = *v20;
            if ( v23 == *v20 )
              goto LABEL_22;
            ++v55;
          }
          if ( !v29 )
            v29 = v20;
          v31 = *(_DWORD *)(a1 + 456);
          ++*(_QWORD *)(a1 + 440);
          v28 = v31 + 1;
          if ( 4 * v28 < 3 * v3 )
          {
            if ( v3 - *(_DWORD *)(a1 + 460) - v28 <= v3 >> 3 )
            {
              v56 = v17;
              sub_A4A350(v57, v3);
              v32 = *(_DWORD *)(a1 + 464);
              if ( !v32 )
              {
LABEL_87:
                ++*(_DWORD *)(a1 + 456);
                BUG();
              }
              v33 = v32 - 1;
              v34 = 0;
              v17 = v56;
              v35 = 1;
              v36 = v33 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
              v37 = *(_QWORD *)(a1 + 448);
              v28 = *(_DWORD *)(a1 + 456) + 1;
              v29 = (_QWORD *)(v37 + 16LL * v36);
              v38 = *v29;
              if ( *v29 != v23 )
              {
                while ( v38 != -4096 )
                {
                  if ( !v34 && v38 == -8192 )
                    v34 = v29;
                  v36 = v33 & (v35 + v36);
                  v29 = (_QWORD *)(v37 + 16LL * v36);
                  v38 = *v29;
                  if ( v23 == *v29 )
                    goto LABEL_30;
                  ++v35;
                }
                if ( v34 )
                  v29 = v34;
              }
            }
            goto LABEL_30;
          }
LABEL_28:
          v54 = v17;
          sub_A4A350(v57, 2 * v3);
          v24 = *(_DWORD *)(a1 + 464);
          if ( !v24 )
            goto LABEL_87;
          v25 = v24 - 1;
          v26 = *(_QWORD *)(a1 + 448);
          v17 = v54;
          v27 = v25 & (((unsigned int)v23 >> 9) ^ ((unsigned int)v23 >> 4));
          v28 = *(_DWORD *)(a1 + 456) + 1;
          v29 = (_QWORD *)(v26 + 16LL * v27);
          v30 = *v29;
          if ( *v29 != v23 )
          {
            v52 = 1;
            v53 = 0;
            while ( v30 != -4096 )
            {
              if ( !v53 && v30 == -8192 )
                v53 = v29;
              v27 = v25 & (v52 + v27);
              v29 = (_QWORD *)(v26 + 16LL * v27);
              v30 = *v29;
              if ( v23 == *v29 )
                goto LABEL_30;
              ++v52;
            }
            if ( v53 )
              v29 = v53;
          }
LABEL_30:
          *(_DWORD *)(a1 + 456) = v28;
          if ( *v29 != -4096 )
            --*(_DWORD *)(a1 + 460);
          *v29 = v23;
          v22 = v29 + 1;
          *((_DWORD *)v29 + 2) = 0;
          goto LABEL_23;
        }
LABEL_22:
        v22 = v20 + 1;
LABEL_23:
        *v22 = v14;
        v16 = *(_QWORD *)(v16 + 8);
        v3 = *(_DWORD *)(a1 + 464);
        if ( v17 == v16 )
          goto LABEL_2;
      }
      ++*(_QWORD *)(a1 + 440);
      goto LABEL_28;
    }
  }
}
