// Function: sub_3530160
// Address: 0x3530160
//
void __fastcall sub_3530160(__int64 *a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 *v3; // r14
  __int64 v5; // r12
  __int64 v6; // r13
  unsigned int v7; // esi
  __int64 v8; // r8
  unsigned int v9; // edi
  _QWORD *v10; // rcx
  __int64 v11; // rdx
  int v12; // r11d
  _QWORD *v13; // r10
  int v14; // ecx
  int v15; // ecx
  int v16; // edx
  int v17; // esi
  __int64 v18; // r8
  unsigned int v19; // edx
  __int64 v20; // rdi
  int v21; // r11d
  _QWORD *v22; // r9
  int v23; // edx
  int v24; // edx
  __int64 v25; // rdi
  _QWORD *v26; // r8
  unsigned int v27; // r15d
  int v28; // r9d
  __int64 v29; // rsi
  __int64 *v30; // [rsp+8h] [rbp-38h]
  __int64 *v31; // [rsp+8h] [rbp-38h]

  v2 = *(__int64 **)(a2 + 112);
  v3 = &v2[*(unsigned int *)(a2 + 120)];
  if ( v2 != v3 )
  {
    while ( 1 )
    {
      v5 = *v2;
      if ( *(_BYTE *)(*v2 + 216) )
        goto LABEL_3;
      v6 = *a1;
      v7 = *(_DWORD *)(*a1 + 24);
      if ( !v7 )
      {
        ++*(_QWORD *)v6;
        goto LABEL_18;
      }
      v8 = *(_QWORD *)(v6 + 8);
      v9 = (v7 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
      v10 = (_QWORD *)(v8 + 8LL * v9);
      v11 = *v10;
      if ( v5 == *v10 )
      {
LABEL_3:
        if ( v3 == ++v2 )
          return;
      }
      else
      {
        v12 = 1;
        v13 = 0;
        while ( v11 != -4096 )
        {
          if ( v13 || v11 != -8192 )
            v10 = v13;
          v9 = (v7 - 1) & (v12 + v9);
          v11 = *(_QWORD *)(v8 + 8LL * v9);
          if ( v5 == v11 )
            goto LABEL_3;
          ++v12;
          v13 = v10;
          v10 = (_QWORD *)(v8 + 8LL * v9);
        }
        if ( !v13 )
          v13 = v10;
        v14 = *(_DWORD *)(v6 + 16);
        ++*(_QWORD *)v6;
        v15 = v14 + 1;
        if ( 4 * v15 < 3 * v7 )
        {
          if ( v7 - *(_DWORD *)(v6 + 20) - v15 <= v7 >> 3 )
          {
            v31 = a1;
            sub_2E52D10(v6, v7);
            v23 = *(_DWORD *)(v6 + 24);
            if ( !v23 )
            {
LABEL_46:
              ++*(_DWORD *)(v6 + 16);
              BUG();
            }
            v24 = v23 - 1;
            v25 = *(_QWORD *)(v6 + 8);
            v26 = 0;
            v27 = v24 & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
            v28 = 1;
            v13 = (_QWORD *)(v25 + 8LL * v27);
            v15 = *(_DWORD *)(v6 + 16) + 1;
            a1 = v31;
            v29 = *v13;
            if ( v5 != *v13 )
            {
              while ( v29 != -4096 )
              {
                if ( !v26 && v29 == -8192 )
                  v26 = v13;
                v27 = v24 & (v28 + v27);
                v13 = (_QWORD *)(v25 + 8LL * v27);
                v29 = *v13;
                if ( v5 == *v13 )
                  goto LABEL_13;
                ++v28;
              }
              if ( v26 )
                v13 = v26;
            }
          }
          goto LABEL_13;
        }
LABEL_18:
        v30 = a1;
        sub_2E52D10(v6, 2 * v7);
        v16 = *(_DWORD *)(v6 + 24);
        if ( !v16 )
          goto LABEL_46;
        v17 = v16 - 1;
        v18 = *(_QWORD *)(v6 + 8);
        v19 = (v16 - 1) & (((unsigned int)v5 >> 9) ^ ((unsigned int)v5 >> 4));
        v13 = (_QWORD *)(v18 + 8LL * v19);
        v15 = *(_DWORD *)(v6 + 16) + 1;
        a1 = v30;
        v20 = *v13;
        if ( v5 != *v13 )
        {
          v21 = 1;
          v22 = 0;
          while ( v20 != -4096 )
          {
            if ( v20 == -8192 && !v22 )
              v22 = v13;
            v19 = v17 & (v21 + v19);
            v13 = (_QWORD *)(v18 + 8LL * v19);
            v20 = *v13;
            if ( v5 == *v13 )
              goto LABEL_13;
            ++v21;
          }
          if ( v22 )
            v13 = v22;
        }
LABEL_13:
        *(_DWORD *)(v6 + 16) = v15;
        if ( *v13 != -4096 )
          --*(_DWORD *)(v6 + 20);
        ++v2;
        *v13 = v5;
        if ( v3 == v2 )
          return;
      }
    }
  }
}
