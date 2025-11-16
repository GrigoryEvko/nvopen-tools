// Function: sub_1B2EB40
// Address: 0x1b2eb40
//
__int64 __fastcall sub_1B2EB40(__int64 a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rcx
  unsigned int v6; // esi
  __int64 *v7; // rdx
  __int64 v8; // r8
  int v10; // edx
  __int64 v11; // r13
  int v12; // r15d
  _QWORD *v13; // rdx
  __int64 i; // rsi
  unsigned int v15; // esi
  int v16; // r11d
  __int64 v17; // r8
  __int64 *v18; // rdx
  unsigned int v19; // edi
  __int64 *v20; // rax
  __int64 v21; // rcx
  int v22; // r9d
  int v23; // eax
  int v24; // ecx
  int v25; // eax
  int v26; // esi
  __int64 v27; // rdi
  unsigned int v28; // eax
  __int64 v29; // r8
  int v30; // r10d
  __int64 *v31; // r9
  int v32; // eax
  int v33; // eax
  __int64 v34; // rdi
  __int64 *v35; // r8
  unsigned int v36; // r14d
  int v37; // r9d
  __int64 v38; // rsi

  v4 = *(unsigned int *)(a1 + 3224);
  if ( (_DWORD)v4 )
  {
    v5 = *(_QWORD *)(a1 + 3208);
    v6 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v7 = (__int64 *)(v5 + 16LL * v6);
    v8 = *v7;
    if ( a2 == *v7 )
    {
LABEL_3:
      if ( v7 != (__int64 *)(v5 + 16 * v4) )
        return *(_QWORD *)(a1 + 112) + 96LL * *((unsigned int *)v7 + 2);
    }
    else
    {
      v10 = 1;
      while ( v8 != -8 )
      {
        v22 = v10 + 1;
        v6 = (v4 - 1) & (v10 + v6);
        v7 = (__int64 *)(v5 + 16LL * v6);
        v8 = *v7;
        if ( a2 == *v7 )
          goto LABEL_3;
        v10 = v22;
      }
    }
  }
  v11 = *(unsigned int *)(a1 + 120);
  v12 = *(_DWORD *)(a1 + 120);
  if ( *(unsigned int *)(a1 + 124) < (unsigned __int64)(v11 + 1) )
    sub_1B2C350((unsigned __int64 **)(a1 + 112), v11 + 1);
  v13 = (_QWORD *)(*(_QWORD *)(a1 + 112) + 96LL * *(unsigned int *)(a1 + 120));
  for ( i = *(_QWORD *)(a1 + 112) + 96 * (v11 + 1); (_QWORD *)i != v13; v13 += 12 )
  {
    if ( v13 )
    {
      memset(v13, 0, 0x60u);
      *((_DWORD *)v13 + 3) = 4;
      *v13 = v13 + 2;
      v13[6] = v13 + 8;
      *((_DWORD *)v13 + 15) = 4;
    }
  }
  v15 = *(_DWORD *)(a1 + 3224);
  *(_DWORD *)(a1 + 120) = v12 + 1;
  if ( !v15 )
  {
    ++*(_QWORD *)(a1 + 3200);
    goto LABEL_33;
  }
  v16 = 1;
  v17 = *(_QWORD *)(a1 + 3208);
  v18 = 0;
  v19 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v20 = (__int64 *)(v17 + 16LL * v19);
  v21 = *v20;
  if ( a2 != *v20 )
  {
    while ( v21 != -8 )
    {
      if ( !v18 && v21 == -16 )
        v18 = v20;
      v19 = (v15 - 1) & (v16 + v19);
      v20 = (__int64 *)(v17 + 16LL * v19);
      v21 = *v20;
      if ( a2 == *v20 )
        goto LABEL_15;
      ++v16;
    }
    if ( !v18 )
      v18 = v20;
    v23 = *(_DWORD *)(a1 + 3216);
    ++*(_QWORD *)(a1 + 3200);
    v24 = v23 + 1;
    if ( 4 * (v23 + 1) < 3 * v15 )
    {
      if ( v15 - *(_DWORD *)(a1 + 3220) - v24 > v15 >> 3 )
      {
LABEL_29:
        *(_DWORD *)(a1 + 3216) = v24;
        if ( *v18 != -8 )
          --*(_DWORD *)(a1 + 3220);
        *v18 = a2;
        *((_DWORD *)v18 + 2) = v12;
        return *(_QWORD *)(a1 + 112) + 96 * v11;
      }
      sub_177C7D0(a1 + 3200, v15);
      v32 = *(_DWORD *)(a1 + 3224);
      if ( v32 )
      {
        v33 = v32 - 1;
        v34 = *(_QWORD *)(a1 + 3208);
        v35 = 0;
        v36 = v33 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v37 = 1;
        v24 = *(_DWORD *)(a1 + 3216) + 1;
        v18 = (__int64 *)(v34 + 16LL * v36);
        v38 = *v18;
        if ( a2 != *v18 )
        {
          while ( v38 != -8 )
          {
            if ( !v35 && v38 == -16 )
              v35 = v18;
            v36 = v33 & (v37 + v36);
            v18 = (__int64 *)(v34 + 16LL * v36);
            v38 = *v18;
            if ( a2 == *v18 )
              goto LABEL_29;
            ++v37;
          }
          if ( v35 )
            v18 = v35;
        }
        goto LABEL_29;
      }
LABEL_56:
      ++*(_DWORD *)(a1 + 3216);
      BUG();
    }
LABEL_33:
    sub_177C7D0(a1 + 3200, 2 * v15);
    v25 = *(_DWORD *)(a1 + 3224);
    if ( v25 )
    {
      v26 = v25 - 1;
      v27 = *(_QWORD *)(a1 + 3208);
      v28 = (v25 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v24 = *(_DWORD *)(a1 + 3216) + 1;
      v18 = (__int64 *)(v27 + 16LL * v28);
      v29 = *v18;
      if ( a2 != *v18 )
      {
        v30 = 1;
        v31 = 0;
        while ( v29 != -8 )
        {
          if ( v29 == -16 && !v31 )
            v31 = v18;
          v28 = v26 & (v30 + v28);
          v18 = (__int64 *)(v27 + 16LL * v28);
          v29 = *v18;
          if ( a2 == *v18 )
            goto LABEL_29;
          ++v30;
        }
        if ( v31 )
          v18 = v31;
      }
      goto LABEL_29;
    }
    goto LABEL_56;
  }
LABEL_15:
  v11 = *((unsigned int *)v20 + 2);
  return *(_QWORD *)(a1 + 112) + 96 * v11;
}
