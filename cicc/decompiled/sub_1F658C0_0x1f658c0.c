// Function: sub_1F658C0
// Address: 0x1f658c0
//
__int64 *__fastcall sub_1F658C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // r10
  unsigned int v9; // esi
  __int64 v10; // r8
  unsigned int v11; // edi
  __int64 *v12; // rax
  __int64 v13; // rcx
  int v14; // r15d
  unsigned int v15; // esi
  __int64 v16; // r9
  __int64 v17; // rdi
  unsigned int v18; // ecx
  __int64 *result; // rax
  __int64 v20; // rdx
  int v21; // r11d
  __int64 *v22; // rdx
  int v23; // eax
  int v24; // ecx
  int v25; // eax
  int v26; // esi
  __int64 v27; // r8
  unsigned int v28; // edx
  int v29; // ecx
  __int64 v30; // rdi
  int v31; // r10d
  __int64 *v32; // r9
  int v33; // r11d
  __int64 *v34; // r10
  int v35; // ecx
  int v36; // eax
  int v37; // esi
  __int64 v38; // rdi
  unsigned int v39; // eax
  __int64 v40; // r8
  int v41; // r10d
  __int64 *v42; // r9
  int v43; // eax
  int v44; // edx
  __int64 v45; // rdi
  int v46; // r9d
  unsigned int v47; // r12d
  __int64 *v48; // r8
  __int64 v49; // rsi
  int v50; // eax
  int v51; // eax
  __int64 v52; // rdi
  int v53; // r9d
  unsigned int v54; // r15d
  __int64 *v55; // r8
  __int64 v56; // rsi
  int v57; // edx
  __int64 *v58; // r11

  v4 = a1 + 64;
  v9 = *(_DWORD *)(a1 + 88);
  if ( !v9 )
  {
    ++*(_QWORD *)(a1 + 64);
    goto LABEL_34;
  }
  v10 = *(_QWORD *)(a1 + 72);
  v11 = (v9 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = (__int64 *)(v10 + 16LL * v11);
  v13 = *v12;
  if ( *v12 == a2 )
  {
    v14 = *((_DWORD *)v12 + 2);
    goto LABEL_4;
  }
  v21 = 1;
  v22 = 0;
  while ( 1 )
  {
    if ( v13 == -8 )
    {
      if ( !v22 )
        v22 = v12;
      v23 = *(_DWORD *)(a1 + 80);
      ++*(_QWORD *)(a1 + 64);
      v24 = v23 + 1;
      if ( 4 * (v23 + 1) < 3 * v9 )
      {
        if ( v9 - *(_DWORD *)(a1 + 84) - v24 > v9 >> 3 )
          goto LABEL_13;
        sub_1F61760(v4, v9);
        v50 = *(_DWORD *)(a1 + 88);
        if ( v50 )
        {
          v51 = v50 - 1;
          v52 = *(_QWORD *)(a1 + 72);
          v53 = 1;
          v54 = v51 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v55 = 0;
          v24 = *(_DWORD *)(a1 + 80) + 1;
          v22 = (__int64 *)(v52 + 16LL * v54);
          v56 = *v22;
          if ( *v22 != a2 )
          {
            while ( v56 != -8 )
            {
              if ( v56 == -16 && !v55 )
                v55 = v22;
              v54 = v51 & (v53 + v54);
              v22 = (__int64 *)(v52 + 16LL * v54);
              v56 = *v22;
              if ( *v22 == a2 )
                goto LABEL_13;
              ++v53;
            }
            if ( v55 )
              v22 = v55;
          }
          goto LABEL_13;
        }
        goto LABEL_85;
      }
LABEL_34:
      sub_1F61760(v4, 2 * v9);
      v36 = *(_DWORD *)(a1 + 88);
      if ( v36 )
      {
        v37 = v36 - 1;
        v38 = *(_QWORD *)(a1 + 72);
        v39 = (v36 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v24 = *(_DWORD *)(a1 + 80) + 1;
        v22 = (__int64 *)(v38 + 16LL * v39);
        v40 = *v22;
        if ( *v22 != a2 )
        {
          v41 = 1;
          v42 = 0;
          while ( v40 != -8 )
          {
            if ( !v42 && v40 == -16 )
              v42 = v22;
            v39 = v37 & (v41 + v39);
            v22 = (__int64 *)(v38 + 16LL * v39);
            v40 = *v22;
            if ( *v22 == a2 )
              goto LABEL_13;
            ++v41;
          }
          if ( v42 )
            v22 = v42;
        }
LABEL_13:
        *(_DWORD *)(a1 + 80) = v24;
        if ( *v22 != -8 )
          --*(_DWORD *)(a1 + 84);
        *v22 = a2;
        v14 = 0;
        v16 = a1 + 96;
        *((_DWORD *)v22 + 2) = 0;
        v15 = *(_DWORD *)(a1 + 120);
        if ( v15 )
          goto LABEL_5;
        goto LABEL_16;
      }
LABEL_85:
      ++*(_DWORD *)(a1 + 80);
      BUG();
    }
    if ( v22 || v13 != -16 )
      v12 = v22;
    v57 = v21 + 1;
    v11 = (v9 - 1) & (v21 + v11);
    v58 = (__int64 *)(v10 + 16LL * v11);
    v13 = *v58;
    if ( *v58 == a2 )
      break;
    v21 = v57;
    v22 = v12;
    v12 = (__int64 *)(v10 + 16LL * v11);
  }
  v14 = *((_DWORD *)v58 + 2);
LABEL_4:
  v15 = *(_DWORD *)(a1 + 120);
  v16 = a1 + 96;
  if ( !v15 )
  {
LABEL_16:
    ++*(_QWORD *)(a1 + 96);
    goto LABEL_17;
  }
LABEL_5:
  v17 = *(_QWORD *)(a1 + 104);
  v18 = (v15 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  result = (__int64 *)(v17 + 24LL * v18);
  v20 = *result;
  if ( *result == a3 )
    goto LABEL_6;
  v33 = 1;
  v34 = 0;
  while ( v20 != -8 )
  {
    if ( !v34 && v20 == -16 )
      v34 = result;
    v18 = (v15 - 1) & (v33 + v18);
    result = (__int64 *)(v17 + 24LL * v18);
    v20 = *result;
    if ( *result == a3 )
      goto LABEL_6;
    ++v33;
  }
  v35 = *(_DWORD *)(a1 + 112);
  if ( v34 )
    result = v34;
  ++*(_QWORD *)(a1 + 96);
  v29 = v35 + 1;
  if ( 4 * v29 >= 3 * v15 )
  {
LABEL_17:
    sub_1F656F0(v16, 2 * v15);
    v25 = *(_DWORD *)(a1 + 120);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 104);
      v28 = (v25 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v29 = *(_DWORD *)(a1 + 112) + 1;
      result = (__int64 *)(v27 + 24LL * v28);
      v30 = *result;
      if ( *result != a3 )
      {
        v31 = 1;
        v32 = 0;
        while ( v30 != -8 )
        {
          if ( !v32 && v30 == -16 )
            v32 = result;
          v28 = v26 & (v31 + v28);
          result = (__int64 *)(v27 + 24LL * v28);
          v30 = *result;
          if ( *result == a3 )
            goto LABEL_30;
          ++v31;
        }
        if ( v32 )
          result = v32;
      }
      goto LABEL_30;
    }
    goto LABEL_84;
  }
  if ( v15 - *(_DWORD *)(a1 + 116) - v29 <= v15 >> 3 )
  {
    sub_1F656F0(v16, v15);
    v43 = *(_DWORD *)(a1 + 120);
    if ( v43 )
    {
      v44 = v43 - 1;
      v45 = *(_QWORD *)(a1 + 104);
      v46 = 1;
      v47 = (v43 - 1) & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
      v48 = 0;
      v29 = *(_DWORD *)(a1 + 112) + 1;
      result = (__int64 *)(v45 + 24LL * v47);
      v49 = *result;
      if ( *result != a3 )
      {
        while ( v49 != -8 )
        {
          if ( v49 == -16 && !v48 )
            v48 = result;
          v47 = v44 & (v46 + v47);
          result = (__int64 *)(v45 + 24LL * v47);
          v49 = *result;
          if ( *result == a3 )
            goto LABEL_30;
          ++v46;
        }
        if ( v48 )
          result = v48;
      }
      goto LABEL_30;
    }
LABEL_84:
    ++*(_DWORD *)(a1 + 112);
    BUG();
  }
LABEL_30:
  *(_DWORD *)(a1 + 112) = v29;
  if ( *result != -8 )
    --*(_DWORD *)(a1 + 116);
  *result = a3;
  *((_DWORD *)result + 2) = 0;
  result[2] = 0;
LABEL_6:
  *((_DWORD *)result + 2) = v14;
  result[2] = a4;
  return result;
}
