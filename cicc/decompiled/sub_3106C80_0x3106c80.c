// Function: sub_3106C80
// Address: 0x3106c80
//
__int64 __fastcall sub_3106C80(__int64 a1)
{
  __int64 v2; // rax
  unsigned int v3; // esi
  unsigned __int64 v4; // rax
  __int64 v5; // r8
  unsigned __int64 *v6; // r10
  int v7; // r11d
  int v8; // r12d
  unsigned int v9; // edi
  unsigned __int64 *v10; // rcx
  unsigned __int64 v11; // rdx
  unsigned __int64 v12; // rax
  unsigned int v13; // esi
  unsigned __int64 v14; // rax
  __int64 v15; // r8
  unsigned __int64 *v16; // r10
  int v17; // r11d
  int v18; // r12d
  unsigned int v19; // edi
  unsigned __int64 *v20; // rcx
  unsigned __int64 v21; // rdx
  int v23; // edx
  int v24; // esi
  __int64 v25; // r8
  unsigned int v26; // edx
  unsigned __int64 v27; // rdi
  int v28; // ecx
  int v29; // edx
  int v30; // esi
  __int64 v31; // r8
  unsigned int v32; // edx
  unsigned __int64 v33; // rdi
  int v34; // ecx
  int v35; // ecx
  int v36; // edx
  int v37; // edx
  __int64 v38; // rdi
  int v39; // r9d
  unsigned int v40; // r12d
  unsigned __int64 *v41; // r8
  unsigned __int64 v42; // rsi
  int v43; // ecx
  int v44; // edx
  int v45; // edx
  __int64 v46; // rdi
  int v47; // r9d
  unsigned int v48; // r12d
  unsigned __int64 *v49; // r8
  unsigned __int64 v50; // rsi
  int v51; // r11d
  unsigned __int64 *v52; // r9
  int v53; // r11d
  unsigned __int64 *v54; // r9
  unsigned __int64 v55; // [rsp+8h] [rbp-28h]
  unsigned __int64 v56; // [rsp+8h] [rbp-28h]
  unsigned __int64 v57; // [rsp+8h] [rbp-28h]
  unsigned __int64 v58; // [rsp+8h] [rbp-28h]

  v2 = sub_31063A0(*(_BYTE **)(a1 + 32), a1, *(unsigned __int8 **)(a1 + 48));
  *(_QWORD *)(a1 + 48) = v2;
  if ( !v2 )
  {
LABEL_4:
    *(_QWORD *)(a1 + 48) = 0;
    v12 = sub_3103B10(*(_BYTE **)(a1 + 32), a1, *(_QWORD *)(a1 + 56));
    *(_QWORD *)(a1 + 56) = v12;
    if ( !v12 )
    {
LABEL_7:
      *(_QWORD *)(a1 + 56) = 0;
      return 0;
    }
    v13 = *(_DWORD *)(a1 + 24);
    v14 = v12 & 0xFFFFFFFFFFFFFFFBLL;
    if ( v13 )
    {
      v15 = *(_QWORD *)(a1 + 8);
      v16 = 0;
      v17 = 1;
      v18 = v14 ^ (v14 >> 9);
      v19 = (v13 - 1) & v18;
      v20 = (unsigned __int64 *)(v15 + 8LL * v19);
      v21 = *v20;
      if ( v14 == *v20 )
        goto LABEL_7;
      while ( v21 != -4 )
      {
        if ( v21 != -16 || v16 )
          v20 = v16;
        v19 = (v13 - 1) & (v17 + v19);
        v21 = *(_QWORD *)(v15 + 8LL * v19);
        if ( v14 == v21 )
          goto LABEL_7;
        ++v17;
        v16 = v20;
        v20 = (unsigned __int64 *)(v15 + 8LL * v19);
      }
      if ( !v16 )
        v16 = v20;
      v43 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)a1;
      v34 = v43 + 1;
      if ( 4 * v34 < 3 * v13 )
      {
        if ( v13 - *(_DWORD *)(a1 + 20) - v34 > v13 >> 3 )
        {
LABEL_17:
          *(_DWORD *)(a1 + 16) = v34;
          if ( *v16 != -4 )
            --*(_DWORD *)(a1 + 20);
          *v16 = v14;
          return *(_QWORD *)(a1 + 56);
        }
        v58 = v14;
        sub_3106460(a1, v13);
        v44 = *(_DWORD *)(a1 + 24);
        if ( v44 )
        {
          v45 = v44 - 1;
          v46 = *(_QWORD *)(a1 + 8);
          v47 = 1;
          v48 = v45 & v18;
          v49 = 0;
          v16 = (unsigned __int64 *)(v46 + 8LL * v48);
          v34 = *(_DWORD *)(a1 + 16) + 1;
          v14 = v58;
          v50 = *v16;
          if ( v58 != *v16 )
          {
            while ( v50 != -4 )
            {
              if ( v50 == -16 && !v49 )
                v49 = v16;
              v48 = v45 & (v47 + v48);
              v16 = (unsigned __int64 *)(v46 + 8LL * v48);
              v50 = *v16;
              if ( v58 == *v16 )
                goto LABEL_17;
              ++v47;
            }
            if ( v49 )
              v16 = v49;
          }
          goto LABEL_17;
        }
LABEL_82:
        ++*(_DWORD *)(a1 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)a1;
    }
    v56 = v14;
    sub_3106460(a1, 2 * v13);
    v29 = *(_DWORD *)(a1 + 24);
    if ( v29 )
    {
      v14 = v56;
      v30 = v29 - 1;
      v31 = *(_QWORD *)(a1 + 8);
      v32 = (v29 - 1) & (v56 ^ (v56 >> 9));
      v16 = (unsigned __int64 *)(v31 + 8LL * v32);
      v33 = *v16;
      v34 = *(_DWORD *)(a1 + 16) + 1;
      if ( v56 != *v16 )
      {
        v53 = 1;
        v54 = 0;
        while ( v33 != -4 )
        {
          if ( !v54 && v33 == -16 )
            v54 = v16;
          v32 = v30 & (v53 + v32);
          v16 = (unsigned __int64 *)(v31 + 8LL * v32);
          v33 = *v16;
          if ( v56 == *v16 )
            goto LABEL_17;
          ++v53;
        }
        if ( v54 )
          v16 = v54;
      }
      goto LABEL_17;
    }
    goto LABEL_82;
  }
  v3 = *(_DWORD *)(a1 + 24);
  v4 = v2 | 4;
  if ( !v3 )
  {
    ++*(_QWORD *)a1;
    goto LABEL_9;
  }
  v5 = *(_QWORD *)(a1 + 8);
  v6 = 0;
  v7 = 1;
  v8 = v4 ^ (v4 >> 9);
  v9 = (v3 - 1) & v8;
  v10 = (unsigned __int64 *)(v5 + 8LL * v9);
  v11 = *v10;
  if ( v4 == *v10 )
    goto LABEL_4;
  while ( v11 != -4 )
  {
    if ( v11 != -16 || v6 )
      v10 = v6;
    v9 = (v3 - 1) & (v7 + v9);
    v11 = *(_QWORD *)(v5 + 8LL * v9);
    if ( v4 == v11 )
      goto LABEL_4;
    ++v7;
    v6 = v10;
    v10 = (unsigned __int64 *)(v5 + 8LL * v9);
  }
  if ( !v6 )
    v6 = v10;
  v35 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v28 = v35 + 1;
  if ( 4 * v28 >= 3 * v3 )
  {
LABEL_9:
    v55 = v4;
    sub_3106460(a1, 2 * v3);
    v23 = *(_DWORD *)(a1 + 24);
    if ( v23 )
    {
      v4 = v55;
      v24 = v23 - 1;
      v25 = *(_QWORD *)(a1 + 8);
      v26 = (v23 - 1) & (v55 ^ (v55 >> 9));
      v6 = (unsigned __int64 *)(v25 + 8LL * v26);
      v27 = *v6;
      v28 = *(_DWORD *)(a1 + 16) + 1;
      if ( v55 != *v6 )
      {
        v51 = 1;
        v52 = 0;
        while ( v27 != -4 )
        {
          if ( v27 == -16 && !v52 )
            v52 = v6;
          v26 = v24 & (v51 + v26);
          v6 = (unsigned __int64 *)(v25 + 8LL * v26);
          v27 = *v6;
          if ( v55 == *v6 )
            goto LABEL_11;
          ++v51;
        }
        if ( v52 )
          v6 = v52;
      }
      goto LABEL_11;
    }
    goto LABEL_83;
  }
  if ( v3 - *(_DWORD *)(a1 + 20) - v28 <= v3 >> 3 )
  {
    v57 = v4;
    sub_3106460(a1, v3);
    v36 = *(_DWORD *)(a1 + 24);
    if ( v36 )
    {
      v37 = v36 - 1;
      v38 = *(_QWORD *)(a1 + 8);
      v39 = 1;
      v40 = v37 & v8;
      v41 = 0;
      v6 = (unsigned __int64 *)(v38 + 8LL * v40);
      v28 = *(_DWORD *)(a1 + 16) + 1;
      v4 = v57;
      v42 = *v6;
      if ( v57 != *v6 )
      {
        while ( v42 != -4 )
        {
          if ( !v41 && v42 == -16 )
            v41 = v6;
          v40 = v37 & (v39 + v40);
          v6 = (unsigned __int64 *)(v38 + 8LL * v40);
          v42 = *v6;
          if ( v57 == *v6 )
            goto LABEL_11;
          ++v39;
        }
        if ( v41 )
          v6 = v41;
      }
      goto LABEL_11;
    }
LABEL_83:
    ++*(_DWORD *)(a1 + 16);
    BUG();
  }
LABEL_11:
  *(_DWORD *)(a1 + 16) = v28;
  if ( *v6 != -4 )
    --*(_DWORD *)(a1 + 20);
  *v6 = v4;
  return *(_QWORD *)(a1 + 48);
}
