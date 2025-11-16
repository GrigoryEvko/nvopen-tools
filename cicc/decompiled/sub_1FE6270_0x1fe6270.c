// Function: sub_1FE6270
// Address: 0x1fe6270
//
__int64 __fastcall sub_1FE6270(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  unsigned __int64 v7; // r12
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned int v11; // ecx
  __int64 *v12; // rdx
  __int64 v13; // r10
  int v15; // edx
  unsigned int v16; // eax
  unsigned int v17; // esi
  unsigned int v18; // r14d
  __int64 v19; // rdi
  unsigned int v20; // ecx
  unsigned __int64 *v21; // rax
  unsigned __int64 v22; // rdx
  int v23; // r11d
  int v24; // r10d
  unsigned __int64 *v25; // r9
  int v26; // edi
  int v27; // ecx
  int v28; // eax
  int v29; // esi
  __int64 v30; // r8
  unsigned int v31; // edx
  unsigned __int64 v32; // rdi
  int v33; // r10d
  unsigned __int64 *v34; // r9
  int v35; // eax
  int v36; // edx
  __int64 v37; // rdi
  unsigned __int64 *v38; // r8
  unsigned int v39; // r15d
  int v40; // r9d
  unsigned __int64 v41; // rsi

  v7 = a2 & 0xFFFFFFFFFFFFFFFBLL;
  v9 = *(unsigned int *)(a1 + 168);
  if ( (_DWORD)v9 )
  {
    v10 = *(_QWORD *)(a1 + 152);
    v11 = (v9 - 1) & (v7 ^ (v7 >> 9));
    v12 = (__int64 *)(v10 + 16LL * v11);
    v13 = *v12;
    if ( v7 == *v12 )
    {
LABEL_3:
      if ( v12 != (__int64 *)(v10 + 16 * v9) )
        return *((unsigned int *)v12 + 2);
    }
    else
    {
      v15 = 1;
      while ( v13 != -4 )
      {
        v23 = v15 + 1;
        v11 = (v9 - 1) & (v15 + v11);
        v12 = (__int64 *)(v10 + 16LL * v11);
        v13 = *v12;
        if ( v7 == *v12 )
          goto LABEL_3;
        v15 = v23;
      }
    }
  }
  v16 = sub_1FE54B0(a1, a3, a4);
  v17 = *(_DWORD *)(a1 + 168);
  v18 = v16;
  if ( !v17 )
  {
    ++*(_QWORD *)(a1 + 144);
    goto LABEL_22;
  }
  v19 = *(_QWORD *)(a1 + 152);
  v20 = (v17 - 1) & (v7 ^ (v7 >> 9));
  v21 = (unsigned __int64 *)(v19 + 16LL * v20);
  v22 = *v21;
  if ( v7 != *v21 )
  {
    v24 = 1;
    v25 = 0;
    while ( v22 != -4 )
    {
      if ( !v25 && v22 == -16 )
        v25 = v21;
      v20 = (v17 - 1) & (v24 + v20);
      v21 = (unsigned __int64 *)(v19 + 16LL * v20);
      v22 = *v21;
      if ( v7 == *v21 )
        goto LABEL_9;
      ++v24;
    }
    v26 = *(_DWORD *)(a1 + 160);
    if ( v25 )
      v21 = v25;
    ++*(_QWORD *)(a1 + 144);
    v27 = v26 + 1;
    if ( 4 * (v26 + 1) < 3 * v17 )
    {
      if ( v17 - *(_DWORD *)(a1 + 164) - v27 > v17 >> 3 )
      {
LABEL_18:
        *(_DWORD *)(a1 + 160) = v27;
        if ( *v21 != -4 )
          --*(_DWORD *)(a1 + 164);
        *v21 = v7;
        *((_DWORD *)v21 + 2) = 0;
        goto LABEL_9;
      }
      sub_1FE5CF0(a1 + 144, v17);
      v35 = *(_DWORD *)(a1 + 168);
      if ( v35 )
      {
        v36 = v35 - 1;
        v37 = *(_QWORD *)(a1 + 152);
        v38 = 0;
        v39 = (v35 - 1) & (v7 ^ (v7 >> 9));
        v40 = 1;
        v27 = *(_DWORD *)(a1 + 160) + 1;
        v21 = (unsigned __int64 *)(v37 + 16LL * v39);
        v41 = *v21;
        if ( v7 != *v21 )
        {
          while ( v41 != -4 )
          {
            if ( !v38 && v41 == -16 )
              v38 = v21;
            v39 = v36 & (v40 + v39);
            v21 = (unsigned __int64 *)(v37 + 16LL * v39);
            v41 = *v21;
            if ( v7 == *v21 )
              goto LABEL_18;
            ++v40;
          }
          if ( v38 )
            v21 = v38;
        }
        goto LABEL_18;
      }
LABEL_50:
      ++*(_DWORD *)(a1 + 160);
      BUG();
    }
LABEL_22:
    sub_1FE5CF0(a1 + 144, 2 * v17);
    v28 = *(_DWORD *)(a1 + 168);
    if ( v28 )
    {
      v29 = v28 - 1;
      v30 = *(_QWORD *)(a1 + 152);
      v27 = *(_DWORD *)(a1 + 160) + 1;
      v31 = (v28 - 1) & (v7 ^ (v7 >> 9));
      v21 = (unsigned __int64 *)(v30 + 16LL * v31);
      v32 = *v21;
      if ( v7 != *v21 )
      {
        v33 = 1;
        v34 = 0;
        while ( v32 != -4 )
        {
          if ( v32 == -16 && !v34 )
            v34 = v21;
          v31 = v29 & (v33 + v31);
          v21 = (unsigned __int64 *)(v30 + 16LL * v31);
          v32 = *v21;
          if ( v7 == *v21 )
            goto LABEL_18;
          ++v33;
        }
        if ( v34 )
          v21 = v34;
      }
      goto LABEL_18;
    }
    goto LABEL_50;
  }
LABEL_9:
  *((_DWORD *)v21 + 2) = v18;
  return v18 | 0x100000000LL;
}
