// Function: sub_25FFEC0
// Address: 0x25ffec0
//
void __fastcall sub_25FFEC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // rbx
  __int64 v6; // rdi
  unsigned int v7; // ecx
  _QWORD *v8; // rdx
  __int64 v9; // rax
  unsigned int v10; // esi
  __int64 v11; // r13
  int v12; // eax
  int v13; // ecx
  __int64 v14; // rdi
  unsigned int v15; // eax
  _QWORD *v16; // r9
  __int64 v17; // rsi
  int v18; // edx
  int v19; // r10d
  _QWORD *v20; // r8
  int v21; // r10d
  int v22; // eax
  int v23; // eax
  int v24; // eax
  __int64 v25; // rsi
  int v26; // r8d
  unsigned int v27; // r14d
  _QWORD *v28; // rdi
  __int64 v29; // rcx

  v3 = *(_QWORD *)(a2 + 8);
  if ( a1 != v3 )
  {
    v4 = a1;
    while ( 1 )
    {
      v10 = *(_DWORD *)(a3 + 24);
      v11 = *(_QWORD *)(*(_QWORD *)(v4 + 16) + 40LL);
      if ( !v10 )
        break;
      v6 = *(_QWORD *)(a3 + 8);
      v7 = (v10 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
      v8 = (_QWORD *)(v6 + 8LL * v7);
      v9 = *v8;
      if ( v11 != *v8 )
      {
        v21 = 1;
        v16 = 0;
        while ( v9 != -4096 )
        {
          if ( v16 || v9 != -8192 )
            v8 = v16;
          v7 = (v10 - 1) & (v21 + v7);
          v9 = *(_QWORD *)(v6 + 8LL * v7);
          if ( v11 == v9 )
            goto LABEL_4;
          ++v21;
          v16 = v8;
          v8 = (_QWORD *)(v6 + 8LL * v7);
        }
        v22 = *(_DWORD *)(a3 + 16);
        if ( !v16 )
          v16 = v8;
        ++*(_QWORD *)a3;
        v18 = v22 + 1;
        if ( 4 * (v22 + 1) < 3 * v10 )
        {
          if ( v10 - *(_DWORD *)(a3 + 20) - v18 <= v10 >> 3 )
          {
            sub_CF28B0(a3, v10);
            v23 = *(_DWORD *)(a3 + 24);
            if ( !v23 )
            {
LABEL_45:
              ++*(_DWORD *)(a3 + 16);
              BUG();
            }
            v24 = v23 - 1;
            v25 = *(_QWORD *)(a3 + 8);
            v26 = 1;
            v27 = v24 & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
            v16 = (_QWORD *)(v25 + 8LL * v27);
            v18 = *(_DWORD *)(a3 + 16) + 1;
            v28 = 0;
            v29 = *v16;
            if ( v11 != *v16 )
            {
              while ( v29 != -4096 )
              {
                if ( v29 == -8192 && !v28 )
                  v28 = v16;
                v27 = v24 & (v26 + v27);
                v16 = (_QWORD *)(v25 + 8LL * v27);
                v29 = *v16;
                if ( v11 == *v16 )
                  goto LABEL_21;
                ++v26;
              }
              if ( v28 )
                v16 = v28;
            }
          }
          goto LABEL_21;
        }
LABEL_7:
        sub_CF28B0(a3, 2 * v10);
        v12 = *(_DWORD *)(a3 + 24);
        if ( !v12 )
          goto LABEL_45;
        v13 = v12 - 1;
        v14 = *(_QWORD *)(a3 + 8);
        v15 = (v12 - 1) & (((unsigned int)v11 >> 9) ^ ((unsigned int)v11 >> 4));
        v16 = (_QWORD *)(v14 + 8LL * v15);
        v17 = *v16;
        v18 = *(_DWORD *)(a3 + 16) + 1;
        if ( v11 != *v16 )
        {
          v19 = 1;
          v20 = 0;
          while ( v17 != -4096 )
          {
            if ( v17 == -8192 && !v20 )
              v20 = v16;
            v15 = v13 & (v19 + v15);
            v16 = (_QWORD *)(v14 + 8LL * v15);
            v17 = *v16;
            if ( v11 == *v16 )
              goto LABEL_21;
            ++v19;
          }
          if ( v20 )
            v16 = v20;
        }
LABEL_21:
        *(_DWORD *)(a3 + 16) = v18;
        if ( *v16 != -4096 )
          --*(_DWORD *)(a3 + 20);
        *v16 = v11;
      }
LABEL_4:
      v4 = *(_QWORD *)(v4 + 8);
      if ( v4 == v3 )
        return;
    }
    ++*(_QWORD *)a3;
    goto LABEL_7;
  }
}
