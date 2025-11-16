// Function: sub_374D810
// Address: 0x374d810
//
__int64 __fastcall sub_374D810(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v5; // rax
  unsigned int v6; // esi
  __int64 v7; // rdi
  int v8; // r11d
  __int64 v9; // r8
  __int64 *v10; // rdx
  unsigned int v11; // ecx
  _QWORD *v12; // rbx
  __int64 v13; // rax
  _DWORD *v14; // rbx
  int v15; // eax
  int v16; // esi
  __int64 v17; // r8
  unsigned int v18; // eax
  int v19; // ecx
  __int64 v20; // rdi
  int v21; // eax
  int v22; // eax
  int v23; // eax
  __int64 v24; // rdi
  __int64 *v25; // r8
  unsigned int v26; // r13d
  int v27; // r9d
  __int64 v28; // rsi
  int v29; // r10d
  __int64 *v30; // r9

  if ( *(_BYTE *)(*(_QWORD *)(a2 + 8) + 8LL) == 11 )
  {
    if ( *(_BYTE *)a2 != 85 )
      return 0;
    v5 = *(_QWORD *)(a2 - 32);
    if ( !v5
      || *(_BYTE *)v5
      || *(_QWORD *)(v5 + 24) != *(_QWORD *)(a2 + 80)
      || (*(_BYTE *)(v5 + 33) & 0x20) == 0
      || (unsigned int)(*(_DWORD *)(v5 + 36) - 142) > 2 )
    {
      return 0;
    }
  }
  v6 = *(_DWORD *)(a1 + 144);
  v7 = a1 + 120;
  if ( !v6 )
  {
    ++*(_QWORD *)(a1 + 120);
    goto LABEL_14;
  }
  v8 = 1;
  v9 = *(_QWORD *)(a1 + 128);
  v10 = 0;
  v11 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v12 = (_QWORD *)(v9 + 16LL * v11);
  v13 = *v12;
  if ( *v12 != a2 )
  {
    while ( v13 != -4096 )
    {
      if ( v13 == -8192 && !v10 )
        v10 = v12;
      v11 = (v6 - 1) & (v8 + v11);
      v12 = (_QWORD *)(v9 + 16LL * v11);
      v13 = *v12;
      if ( *v12 == a2 )
        goto LABEL_11;
      ++v8;
    }
    v21 = *(_DWORD *)(a1 + 136);
    if ( !v10 )
      v10 = v12;
    ++*(_QWORD *)(a1 + 120);
    v19 = v21 + 1;
    if ( 4 * (v21 + 1) < 3 * v6 )
    {
      if ( v6 - *(_DWORD *)(a1 + 140) - v19 > v6 >> 3 )
      {
LABEL_16:
        *(_DWORD *)(a1 + 136) = v19;
        if ( *v10 != -4096 )
          --*(_DWORD *)(a1 + 140);
        *v10 = a2;
        v14 = v10 + 1;
        *((_DWORD *)v10 + 2) = 0;
        goto LABEL_12;
      }
      sub_3384500(v7, v6);
      v22 = *(_DWORD *)(a1 + 144);
      if ( v22 )
      {
        v23 = v22 - 1;
        v24 = *(_QWORD *)(a1 + 128);
        v25 = 0;
        v26 = v23 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
        v27 = 1;
        v19 = *(_DWORD *)(a1 + 136) + 1;
        v10 = (__int64 *)(v24 + 16LL * v26);
        v28 = *v10;
        if ( *v10 != a2 )
        {
          while ( v28 != -4096 )
          {
            if ( v28 == -8192 && !v25 )
              v25 = v10;
            v26 = v23 & (v27 + v26);
            v10 = (__int64 *)(v24 + 16LL * v26);
            v28 = *v10;
            if ( *v10 == a2 )
              goto LABEL_16;
            ++v27;
          }
          if ( v25 )
            v10 = v25;
        }
        goto LABEL_16;
      }
LABEL_50:
      ++*(_DWORD *)(a1 + 136);
      BUG();
    }
LABEL_14:
    sub_3384500(v7, 2 * v6);
    v15 = *(_DWORD *)(a1 + 144);
    if ( v15 )
    {
      v16 = v15 - 1;
      v17 = *(_QWORD *)(a1 + 128);
      v18 = (v15 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v19 = *(_DWORD *)(a1 + 136) + 1;
      v10 = (__int64 *)(v17 + 16LL * v18);
      v20 = *v10;
      if ( *v10 != a2 )
      {
        v29 = 1;
        v30 = 0;
        while ( v20 != -4096 )
        {
          if ( !v30 && v20 == -8192 )
            v30 = v10;
          v18 = v16 & (v29 + v18);
          v10 = (__int64 *)(v17 + 16LL * v18);
          v20 = *v10;
          if ( *v10 == a2 )
            goto LABEL_16;
          ++v29;
        }
        if ( v30 )
          v10 = v30;
      }
      goto LABEL_16;
    }
    goto LABEL_50;
  }
LABEL_11:
  v14 = v12 + 1;
LABEL_12:
  result = sub_374D200((_QWORD *)a1, a2);
  *v14 = result;
  return result;
}
