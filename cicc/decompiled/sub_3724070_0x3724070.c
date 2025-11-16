// Function: sub_3724070
// Address: 0x3724070
//
unsigned __int64 __fastcall sub_3724070(__int64 *a1, __int64 a2, __int64 a3, unsigned int a4, __int64 a5, int a6)
{
  __int64 v6; // rdx
  __int64 v8; // r12
  int v9; // r13d
  unsigned int v10; // esi
  __int64 v11; // r8
  _DWORD *v12; // rdx
  int v13; // r10d
  unsigned int v14; // edi
  _DWORD *v15; // rax
  int v16; // ecx
  int v17; // eax
  int v18; // esi
  __int64 v19; // r8
  unsigned int v20; // eax
  int v21; // ecx
  int v22; // edi
  int v23; // eax
  int v24; // eax
  int v25; // eax
  __int64 v26; // rdi
  _DWORD *v27; // r8
  unsigned int v28; // r14d
  int v29; // r9d
  int v30; // esi
  int v31; // r10d
  _DWORD *v32; // r9

  if ( *(char *)(a2 + 43) < 0 )
  {
    v8 = *a1;
    v9 = *(_DWORD *)(a2 + 44);
    v10 = *(_DWORD *)(*a1 + 24);
    if ( v10 )
    {
      v11 = *(_QWORD *)(v8 + 8);
      v12 = 0;
      v13 = 1;
      v14 = (v10 - 1) & (37 * v9);
      v15 = (_DWORD *)(v11 + 8LL * v14);
      v16 = *v15;
      if ( v9 == *v15 )
      {
LABEL_7:
        a4 = v15[1];
LABEL_8:
        a6 = 2;
        return __PAIR64__(a6, a4);
      }
      while ( v16 != -1 )
      {
        if ( !v12 && v16 == -2 )
          v12 = v15;
        v14 = (v10 - 1) & (v13 + v14);
        v15 = (_DWORD *)(v11 + 8LL * v14);
        v16 = *v15;
        if ( v9 == *v15 )
          goto LABEL_7;
        ++v13;
      }
      if ( !v12 )
        v12 = v15;
      v23 = *(_DWORD *)(v8 + 16);
      ++*(_QWORD *)v8;
      v21 = v23 + 1;
      if ( 4 * (v23 + 1) < 3 * v10 )
      {
        if ( v10 - *(_DWORD *)(v8 + 20) - v21 > v10 >> 3 )
        {
LABEL_12:
          *(_DWORD *)(v8 + 16) = v21;
          if ( *v12 != -1 )
            --*(_DWORD *)(v8 + 20);
          *v12 = v9;
          a4 = 0;
          v12[1] = 0;
          goto LABEL_8;
        }
        sub_A09770(v8, v10);
        v24 = *(_DWORD *)(v8 + 24);
        if ( v24 )
        {
          v25 = v24 - 1;
          v26 = *(_QWORD *)(v8 + 8);
          v27 = 0;
          v28 = v25 & (37 * v9);
          v29 = 1;
          v21 = *(_DWORD *)(v8 + 16) + 1;
          v12 = (_DWORD *)(v26 + 8LL * v28);
          v30 = *v12;
          if ( v9 != *v12 )
          {
            while ( v30 != -1 )
            {
              if ( !v27 && v30 == -2 )
                v27 = v12;
              v28 = v25 & (v29 + v28);
              v12 = (_DWORD *)(v26 + 8LL * v28);
              v30 = *v12;
              if ( v9 == *v12 )
                goto LABEL_12;
              ++v29;
            }
            if ( v27 )
              v12 = v27;
          }
          goto LABEL_12;
        }
LABEL_46:
        ++*(_DWORD *)(v8 + 16);
        BUG();
      }
    }
    else
    {
      ++*(_QWORD *)v8;
    }
    sub_A09770(v8, 2 * v10);
    v17 = *(_DWORD *)(v8 + 24);
    if ( v17 )
    {
      v18 = v17 - 1;
      v19 = *(_QWORD *)(v8 + 8);
      v20 = (v17 - 1) & (37 * v9);
      v21 = *(_DWORD *)(v8 + 16) + 1;
      v12 = (_DWORD *)(v19 + 8LL * v20);
      v22 = *v12;
      if ( v9 != *v12 )
      {
        v31 = 1;
        v32 = 0;
        while ( v22 != -1 )
        {
          if ( !v32 && v22 == -2 )
            v32 = v12;
          v20 = v18 & (v31 + v20);
          v12 = (_DWORD *)(v19 + 8LL * v20);
          v22 = *v12;
          if ( v9 == *v12 )
            goto LABEL_12;
          ++v31;
        }
        if ( v32 )
          v12 = v32;
      }
      goto LABEL_12;
    }
    goto LABEL_46;
  }
  v6 = a1[2];
  if ( *(_DWORD *)(v6 + 8) > 1u )
  {
    a6 = 1;
    a4 = *(_DWORD *)(*(_QWORD *)v6 + 4LL * *(unsigned int *)(a2 + 44));
  }
  return __PAIR64__(a6, a4);
}
