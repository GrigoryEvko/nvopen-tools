// Function: sub_31CE310
// Address: 0x31ce310
//
void __fastcall sub_31CE310(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v4; // r15
  const void *v5; // r14
  __int64 v6; // r8
  int v7; // r10d
  __int64 v8; // rdi
  _QWORD *v9; // r9
  unsigned int v10; // ecx
  _QWORD *v11; // rdx
  __int64 v12; // rax
  unsigned int v13; // esi
  int v14; // eax
  int v15; // ecx
  __int64 v16; // rdi
  unsigned int v17; // eax
  __int64 v18; // rsi
  int v19; // edx
  int v20; // r10d
  int v21; // eax
  __int64 v22; // rax
  int v23; // eax
  int v24; // eax
  __int64 v25; // rsi
  unsigned int v26; // r13d
  _QWORD *v27; // rdi
  __int64 v28; // rcx

  v2 = *(_QWORD *)(a2 + 16);
  if ( v2 )
  {
    v4 = a1 + 272;
    v5 = (const void *)(a1 + 16);
    while ( 1 )
    {
      v13 = *(_DWORD *)(a1 + 296);
      if ( !v13 )
        break;
      v6 = v13 - 1;
      v7 = 1;
      v8 = *(_QWORD *)(a1 + 280);
      v9 = 0;
      v10 = v6 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
      v11 = (_QWORD *)(v8 + 8LL * v10);
      v12 = *v11;
      if ( *v11 == v2 )
      {
LABEL_4:
        v2 = *(_QWORD *)(v2 + 8);
        if ( !v2 )
          return;
      }
      else
      {
        while ( v12 != -4096 )
        {
          if ( v9 || v12 != -8192 )
            v11 = v9;
          v10 = v6 & (v7 + v10);
          v12 = *(_QWORD *)(v8 + 8LL * v10);
          if ( v2 == v12 )
            goto LABEL_4;
          ++v7;
          v9 = v11;
          v11 = (_QWORD *)(v8 + 8LL * v10);
        }
        v21 = *(_DWORD *)(a1 + 288);
        if ( !v9 )
          v9 = v11;
        ++*(_QWORD *)(a1 + 272);
        v19 = v21 + 1;
        if ( 4 * (v21 + 1) < 3 * v13 )
        {
          if ( v13 - *(_DWORD *)(a1 + 292) - v19 <= v13 >> 3 )
          {
            sub_313CFE0(v4, v13);
            v23 = *(_DWORD *)(a1 + 296);
            if ( !v23 )
            {
LABEL_46:
              ++*(_DWORD *)(a1 + 288);
              BUG();
            }
            v24 = v23 - 1;
            v25 = *(_QWORD *)(a1 + 280);
            v6 = 1;
            v26 = v24 & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
            v9 = (_QWORD *)(v25 + 8LL * v26);
            v19 = *(_DWORD *)(a1 + 288) + 1;
            v27 = 0;
            v28 = *v9;
            if ( v2 != *v9 )
            {
              while ( v28 != -4096 )
              {
                if ( v28 == -8192 && !v27 )
                  v27 = v9;
                v26 = v24 & (v6 + v26);
                v9 = (_QWORD *)(v25 + 8LL * v26);
                v28 = *v9;
                if ( v2 == *v9 )
                  goto LABEL_23;
                v6 = (unsigned int)(v6 + 1);
              }
              if ( v27 )
                v9 = v27;
            }
          }
          goto LABEL_23;
        }
LABEL_7:
        sub_313CFE0(v4, 2 * v13);
        v14 = *(_DWORD *)(a1 + 296);
        if ( !v14 )
          goto LABEL_46;
        v15 = v14 - 1;
        v16 = *(_QWORD *)(a1 + 280);
        v17 = (v14 - 1) & (((unsigned int)v2 >> 9) ^ ((unsigned int)v2 >> 4));
        v9 = (_QWORD *)(v16 + 8LL * v17);
        v18 = *v9;
        v19 = *(_DWORD *)(a1 + 288) + 1;
        if ( v2 != *v9 )
        {
          v20 = 1;
          v6 = 0;
          while ( v18 != -4096 )
          {
            if ( !v6 && v18 == -8192 )
              v6 = (__int64)v9;
            v17 = v15 & (v20 + v17);
            v9 = (_QWORD *)(v16 + 8LL * v17);
            v18 = *v9;
            if ( v2 == *v9 )
              goto LABEL_23;
            ++v20;
          }
          if ( v6 )
            v9 = (_QWORD *)v6;
        }
LABEL_23:
        *(_DWORD *)(a1 + 288) = v19;
        if ( *v9 != -4096 )
          --*(_DWORD *)(a1 + 292);
        *v9 = v2;
        v22 = *(unsigned int *)(a1 + 8);
        if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 12) )
        {
          sub_C8D5F0(a1, v5, v22 + 1, 8u, v6, (__int64)v9);
          v22 = *(unsigned int *)(a1 + 8);
        }
        *(_QWORD *)(*(_QWORD *)a1 + 8 * v22) = v2;
        ++*(_DWORD *)(a1 + 8);
        v2 = *(_QWORD *)(v2 + 8);
        if ( !v2 )
          return;
      }
    }
    ++*(_QWORD *)(a1 + 272);
    goto LABEL_7;
  }
}
