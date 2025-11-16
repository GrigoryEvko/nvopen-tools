// Function: sub_1FA7E80
// Address: 0x1fa7e80
//
void __fastcall sub_1FA7E80(__int64 a1, __int64 a2)
{
  __int64 v2; // r15
  __int64 v4; // rbx
  int v5; // r11d
  _QWORD *v6; // r9
  _QWORD *v7; // rdx
  unsigned int v8; // edi
  _QWORD *v9; // rax
  __int64 v10; // rcx
  __int64 v11; // r13
  unsigned int v12; // esi
  int v13; // r8d
  int v14; // eax
  int v15; // edi
  __int64 v16; // rsi
  unsigned int v17; // eax
  int v18; // ecx
  int v19; // r11d
  _QWORD *v20; // r10
  int v21; // eax
  __int64 v22; // rax
  int v23; // eax
  int v24; // eax
  int v25; // r10d
  unsigned int v26; // r14d
  __int64 v27; // rdi
  __int64 v28; // rsi
  int v29; // [rsp-40h] [rbp-40h]
  int v30; // [rsp-40h] [rbp-40h]

  if ( a2 )
  {
    v2 = a1 + 560;
    v4 = a2;
    do
    {
      v11 = *(_QWORD *)(v4 + 16);
      if ( *(_WORD *)(v11 + 24) != 212 )
      {
        v12 = *(_DWORD *)(a1 + 584);
        v13 = *(_DWORD *)(a1 + 40);
        if ( !v12 )
        {
          ++*(_QWORD *)(a1 + 560);
          goto LABEL_8;
        }
        v5 = 1;
        v6 = *(_QWORD **)(a1 + 568);
        v7 = 0;
        v8 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v9 = &v6[2 * v8];
        v10 = *v9;
        if ( v11 != *v9 )
        {
          while ( v10 != -8 )
          {
            if ( v10 != -16 || v7 )
              v9 = v7;
            v8 = (v12 - 1) & (v5 + v8);
            v10 = v6[2 * v8];
            if ( v11 == v10 )
              goto LABEL_4;
            ++v5;
            v7 = v9;
            v9 = &v6[2 * v8];
          }
          if ( !v7 )
            v7 = v9;
          v21 = *(_DWORD *)(a1 + 576);
          ++*(_QWORD *)(a1 + 560);
          v18 = v21 + 1;
          if ( 4 * (v21 + 1) >= 3 * v12 )
          {
LABEL_8:
            v29 = v13;
            sub_1D45DD0(v2, 2 * v12);
            v14 = *(_DWORD *)(a1 + 584);
            if ( !v14 )
              goto LABEL_48;
            v15 = v14 - 1;
            v16 = *(_QWORD *)(a1 + 568);
            v13 = v29;
            v17 = (v14 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
            v18 = *(_DWORD *)(a1 + 576) + 1;
            v7 = (_QWORD *)(v16 + 16LL * v17);
            v6 = (_QWORD *)*v7;
            if ( v11 != *v7 )
            {
              v19 = 1;
              v20 = 0;
              while ( v6 != (_QWORD *)-8LL )
              {
                if ( v6 == (_QWORD *)-16LL && !v20 )
                  v20 = v7;
                v17 = v15 & (v19 + v17);
                v7 = (_QWORD *)(v16 + 16LL * v17);
                v6 = (_QWORD *)*v7;
                if ( v11 == *v7 )
                  goto LABEL_25;
                ++v19;
              }
              if ( v20 )
                v7 = v20;
            }
          }
          else if ( v12 - *(_DWORD *)(a1 + 580) - v18 <= v12 >> 3 )
          {
            v30 = v13;
            sub_1D45DD0(v2, v12);
            v23 = *(_DWORD *)(a1 + 584);
            if ( !v23 )
            {
LABEL_48:
              ++*(_DWORD *)(a1 + 576);
              BUG();
            }
            v24 = v23 - 1;
            v6 = 0;
            v13 = v30;
            v25 = 1;
            v26 = v24 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
            v27 = *(_QWORD *)(a1 + 568);
            v18 = *(_DWORD *)(a1 + 576) + 1;
            v7 = (_QWORD *)(v27 + 16LL * v26);
            v28 = *v7;
            if ( v11 != *v7 )
            {
              while ( v28 != -8 )
              {
                if ( !v6 && v28 == -16 )
                  v6 = v7;
                v26 = v24 & (v25 + v26);
                v7 = (_QWORD *)(v27 + 16LL * v26);
                v28 = *v7;
                if ( v11 == *v7 )
                  goto LABEL_25;
                ++v25;
              }
              if ( v6 )
                v7 = v6;
            }
          }
LABEL_25:
          *(_DWORD *)(a1 + 576) = v18;
          if ( *v7 != -8 )
            --*(_DWORD *)(a1 + 580);
          *v7 = v11;
          *((_DWORD *)v7 + 2) = v13;
          v22 = *(unsigned int *)(a1 + 40);
          if ( (unsigned int)v22 >= *(_DWORD *)(a1 + 44) )
          {
            sub_16CD150(a1 + 32, (const void *)(a1 + 48), 0, 8, v13, (int)v6);
            v22 = *(unsigned int *)(a1 + 40);
          }
          *(_QWORD *)(*(_QWORD *)(a1 + 32) + 8 * v22) = v11;
          ++*(_DWORD *)(a1 + 40);
        }
      }
LABEL_4:
      v4 = *(_QWORD *)(v4 + 32);
    }
    while ( v4 );
  }
}
