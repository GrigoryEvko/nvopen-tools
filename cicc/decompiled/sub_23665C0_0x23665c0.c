// Function: sub_23665C0
// Address: 0x23665c0
//
void __fastcall sub_23665C0(__int64 a1, _DWORD *a2, _DWORD *a3)
{
  _DWORD *v5; // rbx
  __int64 v6; // r8
  unsigned int v7; // edx
  _DWORD *v8; // rdi
  int v9; // ecx
  unsigned int v10; // esi
  int v11; // eax
  int v12; // ecx
  __int64 v13; // r8
  unsigned int v14; // eax
  _DWORD *v15; // r10
  int v16; // edi
  int v17; // edx
  int v18; // r11d
  _DWORD *v19; // r9
  int v20; // r11d
  int v21; // eax
  int v22; // eax
  int v23; // eax
  __int64 v24; // r8
  int v25; // r11d
  unsigned int v26; // edi
  int v27; // esi

  if ( a2 != a3 )
  {
    v5 = a2;
    while ( 1 )
    {
      v10 = *(_DWORD *)(a1 + 24);
      if ( !v10 )
        break;
      v6 = *(_QWORD *)(a1 + 8);
      v7 = (v10 - 1) & (37 * *v5);
      v8 = (_DWORD *)(v6 + 4LL * v7);
      v9 = *v8;
      if ( *v8 != *v5 )
      {
        v20 = 1;
        v15 = 0;
        while ( v9 != -1 )
        {
          if ( v15 || v9 != -2 )
            v8 = v15;
          v7 = (v10 - 1) & (v20 + v7);
          v9 = *(_DWORD *)(v6 + 4LL * v7);
          if ( *v5 == v9 )
            goto LABEL_4;
          ++v20;
          v15 = v8;
          v8 = (_DWORD *)(v6 + 4LL * v7);
        }
        v21 = *(_DWORD *)(a1 + 16);
        if ( !v15 )
          v15 = v8;
        ++*(_QWORD *)a1;
        v17 = v21 + 1;
        if ( 4 * (v21 + 1) < 3 * v10 )
        {
          if ( v10 - *(_DWORD *)(a1 + 20) - v17 <= v10 >> 3 )
          {
            sub_A08C50(a1, v10);
            v22 = *(_DWORD *)(a1 + 24);
            if ( !v22 )
            {
LABEL_43:
              ++*(_DWORD *)(a1 + 16);
              BUG();
            }
            v23 = v22 - 1;
            v24 = *(_QWORD *)(a1 + 8);
            v25 = 1;
            v19 = 0;
            v26 = v23 & (37 * *v5);
            v15 = (_DWORD *)(v24 + 4LL * v26);
            v27 = *v15;
            v17 = *(_DWORD *)(a1 + 16) + 1;
            if ( *v5 != *v15 )
            {
              while ( v27 != -1 )
              {
                if ( !v19 && v27 == -2 )
                  v19 = v15;
                v26 = v23 & (v25 + v26);
                v15 = (_DWORD *)(v24 + 4LL * v26);
                v27 = *v15;
                if ( *v5 == *v15 )
                  goto LABEL_21;
                ++v25;
              }
LABEL_11:
              if ( v19 )
                v15 = v19;
            }
          }
LABEL_21:
          *(_DWORD *)(a1 + 16) = v17;
          if ( *v15 != -1 )
            --*(_DWORD *)(a1 + 20);
          *v15 = *v5;
          goto LABEL_4;
        }
LABEL_7:
        sub_A08C50(a1, 2 * v10);
        v11 = *(_DWORD *)(a1 + 24);
        if ( !v11 )
          goto LABEL_43;
        v12 = v11 - 1;
        v13 = *(_QWORD *)(a1 + 8);
        v14 = (v11 - 1) & (37 * *v5);
        v15 = (_DWORD *)(v13 + 4LL * (v12 & (unsigned int)(37 * *v5)));
        v16 = *v15;
        v17 = *(_DWORD *)(a1 + 16) + 1;
        if ( *v15 != *v5 )
        {
          v18 = 1;
          v19 = 0;
          while ( v16 != -1 )
          {
            if ( v16 == -2 && !v19 )
              v19 = v15;
            v14 = v12 & (v18 + v14);
            v15 = (_DWORD *)(v13 + 4LL * v14);
            v16 = *v15;
            if ( *v5 == *v15 )
              goto LABEL_21;
            ++v18;
          }
          goto LABEL_11;
        }
        goto LABEL_21;
      }
LABEL_4:
      if ( a3 == ++v5 )
        return;
    }
    ++*(_QWORD *)a1;
    goto LABEL_7;
  }
}
