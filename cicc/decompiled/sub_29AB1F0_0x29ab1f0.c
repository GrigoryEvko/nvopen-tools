// Function: sub_29AB1F0
// Address: 0x29ab1f0
//
__int64 __fastcall sub_29AB1F0(__int64 a1)
{
  __int64 v1; // r9
  __int64 *v2; // rcx
  __int64 v3; // rax
  __int64 *v4; // r11
  __int64 *i; // rcx
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // rsi
  int v9; // esi
  __int64 v10; // rsi
  __int64 v11; // r9
  int v12; // esi
  __int64 v13; // r10
  int v14; // r8d
  unsigned int v15; // esi
  __int64 v16; // rbx
  int v17; // r12d
  __int64 v19; // rsi
  __int64 v20; // r10
  int v21; // r8d
  __int64 v22; // rbx
  int v23; // r9d
  unsigned int v24; // r8d
  __int64 v25; // r12
  __int64 v26; // r8
  int v27; // r13d
  __int64 v28; // r11
  __int64 v29; // rsi
  __int64 v30; // r11
  int v31; // ebx
  __int64 v32; // r14
  int v33; // r13d
  unsigned int v34; // r8d
  __int64 v35; // r10
  __int64 v36; // rdx
  __int64 v37; // r10
  __int64 v38; // rdx
  __int64 v39; // r8
  int v40; // r15d
  __int64 v41; // [rsp-8h] [rbp-8h]

  v1 = *(unsigned int *)(a1 + 96);
  if ( !(_DWORD)v1 )
    return 0;
  v2 = *(__int64 **)(a1 + 88);
  v3 = *v2;
  if ( *(_BYTE *)(a1 + 48) )
  {
    v28 = *(_QWORD *)(v3 + 72);
    if ( *(_DWORD *)(*(_QWORD *)(v28 + 24) + 8LL) >> 8 )
    {
      v29 = *(_QWORD *)(v28 + 80);
      v30 = v28 + 72;
      if ( v29 != v30 )
      {
        v31 = *(_DWORD *)(a1 + 80);
        v32 = *(_QWORD *)(a1 + 64);
        v33 = v31 - 1;
        while ( 1 )
        {
          v36 = v29 - 24;
          if ( !v29 )
            v36 = 0;
          if ( !v31 )
            goto LABEL_43;
          v34 = v33 & (((unsigned int)v36 >> 9) ^ ((unsigned int)v36 >> 4));
          v35 = *(_QWORD *)(v32 + 8LL * v34);
          if ( v36 != v35 )
            break;
LABEL_39:
          v29 = *(_QWORD *)(v29 + 8);
          if ( v30 == v29 )
            goto LABEL_3;
        }
        v40 = 1;
        while ( v35 != -4096 )
        {
          v34 = v33 & (v40 + v34);
          *((_DWORD *)&v41 - 11) = v40 + 1;
          v35 = *(_QWORD *)(v32 + 8LL * v34);
          if ( v36 == v35 )
            goto LABEL_39;
          v40 = *((_DWORD *)&v41 - 11);
        }
LABEL_43:
        v37 = v36 + 48;
        v38 = *(_QWORD *)(v36 + 56);
        if ( v37 != v38 )
        {
          while ( 1 )
          {
            if ( !v38 )
              BUG();
            if ( *(_BYTE *)(v38 - 24) == 85 )
            {
              v39 = *(_QWORD *)(v38 - 56);
              if ( v39 )
              {
                if ( !*(_BYTE *)v39
                  && *(_QWORD *)(v39 + 24) == *(_QWORD *)(v38 + 56)
                  && (unsigned int)(*(_DWORD *)(v39 + 36) - 374) <= 1 )
                {
                  break;
                }
              }
            }
            v38 = *(_QWORD *)(v38 + 8);
            if ( v37 == v38 )
              goto LABEL_39;
          }
          if ( v37 != v38 )
            return 0;
        }
        goto LABEL_39;
      }
    }
  }
LABEL_3:
  v4 = &v2[v1];
  for ( i = v2 + 1; ; ++i )
  {
    v6 = *(_QWORD *)(v3 + 56);
    v7 = v3 + 48;
    if ( v6 != v7 )
      break;
LABEL_28:
    if ( v4 == i )
      return 1;
    v3 = *i;
  }
  while ( 1 )
  {
    if ( !v6 )
      BUG();
    if ( *(_BYTE *)(v6 - 24) != 85 )
      goto LABEL_6;
    v8 = *(_QWORD *)(v6 - 56);
    if ( !v8 || *(_BYTE *)v8 || *(_QWORD *)(v8 + 24) != *(_QWORD *)(v6 + 56) || (*(_BYTE *)(v8 + 33) & 0x20) == 0 )
      goto LABEL_6;
    v9 = *(_DWORD *)(v8 + 36);
    if ( v9 == 343 )
      break;
    if ( v9 == 342 )
    {
      v10 = *(_QWORD *)(v6 - 32LL * (*(_DWORD *)(v6 - 20) & 0x7FFFFFF) - 24);
      if ( *(_BYTE *)v10 <= 0x1Cu )
        return 0;
      v11 = *(_QWORD *)(v10 + 40);
      v12 = *(_DWORD *)(a1 + 80);
      v13 = *(_QWORD *)(a1 + 64);
      if ( !v12 )
        return 0;
      v14 = v12 - 1;
      v15 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v16 = *(_QWORD *)(v13 + 8LL * v15);
      if ( v11 != v16 )
      {
        v17 = 1;
        while ( v16 != -4096 )
        {
          v15 = v14 & (v17 + v15);
          v16 = *(_QWORD *)(v13 + 8LL * v15);
          if ( v11 == v16 )
            goto LABEL_6;
          ++v17;
        }
        return 0;
      }
    }
LABEL_6:
    v6 = *(_QWORD *)(v6 + 8);
    if ( v7 == v6 )
      goto LABEL_28;
  }
  v19 = *(_QWORD *)(v6 - 8);
  if ( !v19 )
    goto LABEL_6;
  while ( 1 )
  {
    v26 = *(_QWORD *)(v19 + 24);
    if ( *(_BYTE *)v26 <= 0x1Cu )
      return 0;
    v20 = *(_QWORD *)(v26 + 40);
    v21 = *(_DWORD *)(a1 + 80);
    v22 = *(_QWORD *)(a1 + 64);
    if ( !v21 )
      return 0;
    v23 = v21 - 1;
    v24 = (v21 - 1) & (((unsigned int)v20 >> 9) ^ ((unsigned int)v20 >> 4));
    v25 = *(_QWORD *)(v22 + 8LL * v24);
    if ( v20 != v25 )
    {
      v27 = 1;
      while ( v25 != -4096 )
      {
        v24 = v23 & (v27 + v24);
        v25 = *(_QWORD *)(v22 + 8LL * v24);
        if ( v20 == v25 )
          goto LABEL_25;
        ++v27;
      }
      return 0;
    }
LABEL_25:
    v19 = *(_QWORD *)(v19 + 8);
    if ( !v19 )
      goto LABEL_6;
  }
}
