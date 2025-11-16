// Function: sub_1A6F5D0
// Address: 0x1a6f5d0
//
__int64 __fastcall sub_1A6F5D0(__int64 a1, __int64 *a2)
{
  __int64 v2; // r8
  __int64 v4; // rbx
  unsigned int v5; // esi
  unsigned __int64 v6; // rbx
  __int64 v7; // r9
  unsigned int v8; // edi
  __int64 *v9; // rcx
  __int64 v10; // rdx
  __int64 *v11; // rdx
  __int64 *v12; // r12
  __int64 v13; // rax
  __int64 *v14; // rbx
  int v16; // r11d
  unsigned __int64 *v17; // rax
  int v18; // edi
  int v19; // ecx
  unsigned int v20; // r8d
  unsigned int v21; // eax
  int v22; // r9d
  int v23; // r9d
  __int64 v24; // r10
  unsigned int v25; // edx
  unsigned __int64 v26; // r8
  int v27; // edi
  unsigned __int64 *v28; // rsi
  int v29; // r8d
  int v30; // r8d
  __int64 v31; // r9
  unsigned __int64 *v32; // rdx
  __int64 v33; // r12
  int v34; // esi
  unsigned __int64 v35; // rdi

  v2 = a1 + 504;
  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 528);
  v6 = v4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v5 )
  {
    v7 = *(_QWORD *)(a1 + 512);
    v8 = (v5 - 1) & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
    v9 = (__int64 *)(v7 + 40LL * v8);
    v10 = *v9;
    if ( v6 == *v9 )
    {
LABEL_3:
      if ( *(_QWORD *)(a1 + 760) )
      {
        if ( *((_DWORD *)v9 + 6) )
        {
          v11 = (__int64 *)v9[2];
          v12 = &v11[2 * *((unsigned int *)v9 + 8)];
          if ( v11 != v12 )
          {
            while ( 1 )
            {
              v13 = *v11;
              v14 = v11;
              if ( *v11 != -8 && *v11 != -16 )
                break;
              v11 += 2;
              if ( v12 == v11 )
                return 0;
            }
            if ( v12 != v11 )
            {
              v20 = 0;
              while ( *(_QWORD *)(a1 + 168) == v14[1] )
              {
                if ( !(_BYTE)v20 )
                {
                  LOBYTE(v21) = sub_15CC8F0(*(_QWORD *)(a1 + 216), v13, **(_QWORD **)(a1 + 760) & 0xFFFFFFFFFFFFFFF8LL);
                  v20 = v21;
                }
                v14 += 2;
                if ( v14 != v12 )
                {
                  while ( 1 )
                  {
                    v13 = *v14;
                    if ( *v14 != -16 && v13 != -8 )
                      break;
                    v14 += 2;
                    if ( v12 == v14 )
                      return v20;
                  }
                  if ( v14 != v12 )
                    continue;
                }
                return v20;
              }
            }
          }
        }
        return 0;
      }
      return 1;
    }
    v16 = 1;
    v17 = 0;
    while ( v10 != -8 )
    {
      if ( v10 == -16 && !v17 )
        v17 = (unsigned __int64 *)v9;
      v8 = (v5 - 1) & (v16 + v8);
      v9 = (__int64 *)(v7 + 40LL * v8);
      v10 = *v9;
      if ( v6 == *v9 )
        goto LABEL_3;
      ++v16;
    }
    v18 = *(_DWORD *)(a1 + 520);
    if ( !v17 )
      v17 = (unsigned __int64 *)v9;
    ++*(_QWORD *)(a1 + 504);
    v19 = v18 + 1;
    if ( 4 * (v18 + 1) < 3 * v5 )
    {
      if ( v5 - *(_DWORD *)(a1 + 524) - v19 > v5 >> 3 )
        goto LABEL_15;
      sub_1A6F390(v2, v5);
      v29 = *(_DWORD *)(a1 + 528);
      if ( v29 )
      {
        v30 = v29 - 1;
        v31 = *(_QWORD *)(a1 + 512);
        v32 = 0;
        LODWORD(v33) = v30 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
        v19 = *(_DWORD *)(a1 + 520) + 1;
        v34 = 1;
        v17 = (unsigned __int64 *)(v31 + 40LL * (unsigned int)v33);
        v35 = *v17;
        if ( v6 != *v17 )
        {
          while ( v35 != -8 )
          {
            if ( v35 == -16 && !v32 )
              v32 = v17;
            v33 = v30 & (unsigned int)(v33 + v34);
            v17 = (unsigned __int64 *)(v31 + 40 * v33);
            v35 = *v17;
            if ( v6 == *v17 )
              goto LABEL_15;
            ++v34;
          }
          if ( v32 )
            v17 = v32;
        }
        goto LABEL_15;
      }
LABEL_61:
      ++*(_DWORD *)(a1 + 520);
      BUG();
    }
  }
  else
  {
    ++*(_QWORD *)(a1 + 504);
  }
  sub_1A6F390(v2, 2 * v5);
  v22 = *(_DWORD *)(a1 + 528);
  if ( !v22 )
    goto LABEL_61;
  v23 = v22 - 1;
  v24 = *(_QWORD *)(a1 + 512);
  v25 = v23 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v19 = *(_DWORD *)(a1 + 520) + 1;
  v17 = (unsigned __int64 *)(v24 + 40LL * v25);
  v26 = *v17;
  if ( v6 != *v17 )
  {
    v27 = 1;
    v28 = 0;
    while ( v26 != -8 )
    {
      if ( !v28 && v26 == -16 )
        v28 = v17;
      v25 = v23 & (v27 + v25);
      v17 = (unsigned __int64 *)(v24 + 40LL * v25);
      v26 = *v17;
      if ( v6 == *v17 )
        goto LABEL_15;
      ++v27;
    }
    if ( v28 )
      v17 = v28;
  }
LABEL_15:
  *(_DWORD *)(a1 + 520) = v19;
  if ( *v17 != -8 )
    --*(_DWORD *)(a1 + 524);
  *v17 = v6;
  v17[1] = 0;
  v17[2] = 0;
  v17[3] = 0;
  *((_DWORD *)v17 + 8) = 0;
  return !*(_QWORD *)(a1 + 760);
}
