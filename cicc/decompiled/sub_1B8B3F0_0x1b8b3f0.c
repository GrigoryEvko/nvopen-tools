// Function: sub_1B8B3F0
// Address: 0x1b8b3f0
//
__int64 __fastcall sub_1B8B3F0(__int64 **a1, __int64 a2)
{
  __int64 v3; // r12
  unsigned int v4; // esi
  __int64 v5; // rcx
  unsigned int v6; // edi
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // r8
  __int64 result; // rax
  __int64 v11; // r8
  __int64 v12; // rax
  unsigned int v13; // edx
  unsigned int v14; // r9d
  __int64 *v15; // rdx
  __int64 v16; // r8
  int v17; // eax
  int v18; // edx
  int v19; // esi
  __int64 v20; // r9
  unsigned int v21; // ecx
  int v22; // edi
  __int64 v23; // r8
  int v24; // r11d
  __int64 *v25; // r10
  int v26; // ecx
  int v27; // edx
  int v28; // ecx
  __int64 v29; // r8
  __int64 *v30; // r9
  unsigned int v31; // r13d
  int v32; // r10d
  __int64 v33; // rsi
  int v34; // r9d
  int v35; // r11d
  __int64 *v36; // r10
  unsigned int v37; // [rsp+Ch] [rbp-24h]
  unsigned int v38; // [rsp+Ch] [rbp-24h]

  v3 = **a1;
  v4 = *(_DWORD *)(v3 + 24);
  v5 = *(_QWORD *)(v3 + 8);
  if ( !v4 )
  {
    v11 = a2 + 24;
    v12 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL);
    if ( v12 == a2 + 24 )
    {
      LODWORD(result) = 0;
      goto LABEL_17;
    }
LABEL_6:
    v13 = 0;
    do
    {
      v12 = *(_QWORD *)(v12 + 8);
      ++v13;
    }
    while ( v12 != v11 );
    result = v13;
    if ( v4 )
    {
      v6 = v4 - 1;
      goto LABEL_10;
    }
LABEL_17:
    ++*(_QWORD *)v3;
    goto LABEL_18;
  }
  v6 = v4 - 1;
  v7 = (v4 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( *v8 == a2 )
  {
LABEL_3:
    if ( v8 != (__int64 *)(v5 + 16LL * v4) )
      return *((unsigned int *)v8 + 2);
  }
  else
  {
    v17 = 1;
    while ( v9 != -8 )
    {
      v34 = v17 + 1;
      v7 = v6 & (v17 + v7);
      v8 = (__int64 *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( *v8 == a2 )
        goto LABEL_3;
      v17 = v34;
    }
  }
  v11 = a2 + 24;
  v12 = *(_QWORD *)(*(_QWORD *)(a2 + 40) + 48LL);
  if ( v12 != a2 + 24 )
    goto LABEL_6;
  result = 0;
LABEL_10:
  v14 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v15 = (__int64 *)(v5 + 16LL * v14);
  v16 = *v15;
  if ( *v15 != a2 )
  {
    v24 = 1;
    v25 = 0;
    while ( v16 != -8 )
    {
      if ( v16 == -16 && !v25 )
        v25 = v15;
      v14 = v6 & (v24 + v14);
      v15 = (__int64 *)(v5 + 16LL * v14);
      v16 = *v15;
      if ( *v15 == a2 )
        goto LABEL_11;
      ++v24;
    }
    v26 = *(_DWORD *)(v3 + 16);
    if ( v25 )
      v15 = v25;
    ++*(_QWORD *)v3;
    v22 = v26 + 1;
    if ( 4 * (v26 + 1) < 3 * v4 )
    {
      if ( v4 - *(_DWORD *)(v3 + 20) - v22 > v4 >> 3 )
      {
LABEL_20:
        *(_DWORD *)(v3 + 16) = v22;
        if ( *v15 != -8 )
          --*(_DWORD *)(v3 + 20);
        *v15 = a2;
        *((_DWORD *)v15 + 2) = 0;
        goto LABEL_11;
      }
      v38 = result;
      sub_1985B20(v3, v4);
      v27 = *(_DWORD *)(v3 + 24);
      if ( v27 )
      {
        v28 = v27 - 1;
        v29 = *(_QWORD *)(v3 + 8);
        v30 = 0;
        v31 = (v27 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v32 = 1;
        v22 = *(_DWORD *)(v3 + 16) + 1;
        result = v38;
        v15 = (__int64 *)(v29 + 16LL * v31);
        v33 = *v15;
        if ( *v15 != a2 )
        {
          while ( v33 != -8 )
          {
            if ( v33 == -16 && !v30 )
              v30 = v15;
            v31 = v28 & (v32 + v31);
            v15 = (__int64 *)(v29 + 16LL * v31);
            v33 = *v15;
            if ( *v15 == a2 )
              goto LABEL_20;
            ++v32;
          }
          if ( v30 )
            v15 = v30;
        }
        goto LABEL_20;
      }
LABEL_57:
      ++*(_DWORD *)(v3 + 16);
      BUG();
    }
LABEL_18:
    v37 = result;
    sub_1985B20(v3, 2 * v4);
    v18 = *(_DWORD *)(v3 + 24);
    if ( v18 )
    {
      v19 = v18 - 1;
      v20 = *(_QWORD *)(v3 + 8);
      v21 = (v18 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v22 = *(_DWORD *)(v3 + 16) + 1;
      result = v37;
      v15 = (__int64 *)(v20 + 16LL * v21);
      v23 = *v15;
      if ( *v15 != a2 )
      {
        v35 = 1;
        v36 = 0;
        while ( v23 != -8 )
        {
          if ( !v36 && v23 == -16 )
            v36 = v15;
          v21 = v19 & (v35 + v21);
          v15 = (__int64 *)(v20 + 16LL * v21);
          v23 = *v15;
          if ( *v15 == a2 )
            goto LABEL_20;
          ++v35;
        }
        if ( v36 )
          v15 = v36;
      }
      goto LABEL_20;
    }
    goto LABEL_57;
  }
LABEL_11:
  *((_DWORD *)v15 + 2) = result;
  return result;
}
