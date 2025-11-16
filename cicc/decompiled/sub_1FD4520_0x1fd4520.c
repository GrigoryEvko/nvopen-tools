// Function: sub_1FD4520
// Address: 0x1fd4520
//
__int64 __fastcall sub_1FD4520(__int64 a1, __int64 *a2)
{
  __int64 v3; // rsi
  __int64 v5; // rdi
  unsigned int v6; // r8d
  __int64 v7; // rcx
  unsigned int v8; // edx
  __int64 v9; // r13
  __int64 *v10; // rax
  __int64 result; // rax
  int v12; // eax
  int v13; // ecx
  __int64 v14; // rdi
  unsigned int v15; // eax
  int v16; // edx
  __int64 *v17; // rsi
  int v18; // r11d
  __int64 v19; // r10
  int v20; // eax
  int v21; // eax
  int v22; // eax
  __int64 v23; // rsi
  int v24; // r8d
  __int64 v25; // rdi
  unsigned int v26; // r14d
  __int64 *v27; // rcx
  int v28; // r9d
  __int64 v29; // r8

  v3 = *a2;
  if ( *(_BYTE *)(v3 + 8) != 10 )
  {
    v5 = a1 + 208;
    v6 = *(_DWORD *)(a1 + 232);
    if ( v6 )
    {
      v7 = *(_QWORD *)(a1 + 216);
      v8 = (v6 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v9 = v7 + 16LL * v8;
      v10 = *(__int64 **)v9;
      if ( *(__int64 **)v9 == a2 )
      {
LABEL_4:
        result = sub_1FDE000(a1, v3);
        *(_DWORD *)(v9 + 8) = result;
        return result;
      }
      v18 = 1;
      v19 = 0;
      while ( v10 != (__int64 *)-8LL )
      {
        if ( !v19 && v10 == (__int64 *)-16LL )
          v19 = v9;
        v8 = (v6 - 1) & (v18 + v8);
        v9 = v7 + 16LL * v8;
        v10 = *(__int64 **)v9;
        if ( *(__int64 **)v9 == a2 )
          goto LABEL_4;
        ++v18;
      }
      v20 = *(_DWORD *)(a1 + 224);
      if ( v19 )
        v9 = v19;
      ++*(_QWORD *)(a1 + 208);
      v16 = v20 + 1;
      if ( 4 * (v20 + 1) < 3 * v6 )
      {
        if ( v6 - *(_DWORD *)(a1 + 228) - v16 > v6 >> 3 )
        {
LABEL_9:
          *(_DWORD *)(a1 + 224) = v16;
          if ( *(_QWORD *)v9 != -8 )
            --*(_DWORD *)(a1 + 228);
          *(_QWORD *)v9 = a2;
          *(_DWORD *)(v9 + 8) = 0;
          v3 = *a2;
          goto LABEL_4;
        }
        sub_1542080(v5, v6);
        v21 = *(_DWORD *)(a1 + 232);
        if ( v21 )
        {
          v22 = v21 - 1;
          v23 = *(_QWORD *)(a1 + 216);
          v24 = 1;
          v25 = 0;
          v26 = v22 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
          v16 = *(_DWORD *)(a1 + 224) + 1;
          v9 = v23 + 16LL * v26;
          v27 = *(__int64 **)v9;
          if ( *(__int64 **)v9 != a2 )
          {
            while ( v27 != (__int64 *)-8LL )
            {
              if ( !v25 && v27 == (__int64 *)-16LL )
                v25 = v9;
              v26 = v22 & (v24 + v26);
              v9 = v23 + 16LL * v26;
              v27 = *(__int64 **)v9;
              if ( *(__int64 **)v9 == a2 )
                goto LABEL_9;
              ++v24;
            }
            if ( v25 )
              v9 = v25;
          }
          goto LABEL_9;
        }
LABEL_44:
        ++*(_DWORD *)(a1 + 224);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)(a1 + 208);
    }
    sub_1542080(v5, 2 * v6);
    v12 = *(_DWORD *)(a1 + 232);
    if ( v12 )
    {
      v13 = v12 - 1;
      v14 = *(_QWORD *)(a1 + 216);
      v15 = (v12 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = *(_DWORD *)(a1 + 224) + 1;
      v9 = v14 + 16LL * v15;
      v17 = *(__int64 **)v9;
      if ( *(__int64 **)v9 != a2 )
      {
        v28 = 1;
        v29 = 0;
        while ( v17 != (__int64 *)-8LL )
        {
          if ( !v29 && v17 == (__int64 *)-16LL )
            v29 = v9;
          v15 = v13 & (v28 + v15);
          v9 = v14 + 16LL * v15;
          v17 = *(__int64 **)v9;
          if ( *(__int64 **)v9 == a2 )
            goto LABEL_9;
          ++v28;
        }
        if ( v29 )
          v9 = v29;
      }
      goto LABEL_9;
    }
    goto LABEL_44;
  }
  return 0;
}
