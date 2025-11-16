// Function: sub_2BFDBB0
// Address: 0x2bfdbb0
//
__int64 __fastcall sub_2BFDBB0(__int64 a1, __int64 a2)
{
  __int64 v4; // r8
  unsigned int v5; // r11d
  int i; // ebx
  unsigned int v7; // esi
  __int64 v8; // r13
  __int64 v9; // r9
  unsigned int v10; // edi
  _QWORD *v11; // rax
  __int64 v12; // rcx
  __int64 *v13; // rax
  __int64 *v14; // rax
  int v15; // ecx
  int v16; // ecx
  __int64 v17; // r9
  unsigned int v18; // esi
  int v19; // eax
  _QWORD *v20; // rdx
  __int64 v21; // rdi
  int v22; // r14d
  _QWORD *v23; // r10
  int v25; // eax
  int v26; // ecx
  int v27; // ecx
  __int64 v28; // rdi
  _QWORD *v29; // r9
  unsigned int v30; // r14d
  int v31; // r10d
  __int64 v32; // rsi
  __int64 v33; // [rsp+0h] [rbp-40h]
  __int64 v34; // [rsp+0h] [rbp-40h]
  unsigned int v35; // [rsp+Ch] [rbp-34h]
  int v36; // [rsp+Ch] [rbp-34h]
  unsigned int v37; // [rsp+Ch] [rbp-34h]

  v4 = sub_2BFD6A0(a1, **(_QWORD **)(a2 + 48));
  if ( (unsigned int)(*(_DWORD *)(a2 + 56) + 1) >> 1 != 1 )
  {
    v5 = (unsigned int)(*(_DWORD *)(a2 + 56) + 1) >> 1;
    for ( i = 1; v5 != i; ++i )
    {
      v14 = *(__int64 **)(a2 + 48);
      if ( i )
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = v14[2 * i - (*(_DWORD *)(a2 + 56) & 1u)];
        if ( !v7 )
        {
LABEL_9:
          ++*(_QWORD *)a1;
          goto LABEL_10;
        }
      }
      else
      {
        v7 = *(_DWORD *)(a1 + 24);
        v8 = *v14;
        if ( !v7 )
          goto LABEL_9;
      }
      v9 = *(_QWORD *)(a1 + 8);
      v10 = (v7 - 1) & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
      v11 = (_QWORD *)(v9 + 16LL * v10);
      v12 = *v11;
      if ( *v11 != v8 )
      {
        v36 = 1;
        v20 = 0;
        while ( v12 != -4096 )
        {
          if ( !v20 && v12 == -8192 )
            v20 = v11;
          v10 = (v7 - 1) & (v36 + v10);
          v11 = (_QWORD *)(v9 + 16LL * v10);
          v12 = *v11;
          if ( v8 == *v11 )
            goto LABEL_5;
          ++v36;
        }
        if ( !v20 )
          v20 = v11;
        v25 = *(_DWORD *)(a1 + 16);
        ++*(_QWORD *)a1;
        v19 = v25 + 1;
        if ( 4 * v19 >= 3 * v7 )
        {
LABEL_10:
          v33 = v4;
          v35 = v5;
          sub_2BFD020(a1, 2 * v7);
          v15 = *(_DWORD *)(a1 + 24);
          if ( !v15 )
            goto LABEL_48;
          v16 = v15 - 1;
          v17 = *(_QWORD *)(a1 + 8);
          v5 = v35;
          v4 = v33;
          v18 = v16 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v19 = *(_DWORD *)(a1 + 16) + 1;
          v20 = (_QWORD *)(v17 + 16LL * v18);
          v21 = *v20;
          if ( v8 != *v20 )
          {
            v22 = 1;
            v23 = 0;
            while ( v21 != -4096 )
            {
              if ( !v23 && v21 == -8192 )
                v23 = v20;
              v18 = v16 & (v22 + v18);
              v20 = (_QWORD *)(v17 + 16LL * v18);
              v21 = *v20;
              if ( v8 == *v20 )
                goto LABEL_24;
              ++v22;
            }
            if ( v23 )
              v20 = v23;
          }
        }
        else if ( v7 - *(_DWORD *)(a1 + 20) - v19 <= v7 >> 3 )
        {
          v34 = v4;
          v37 = v5;
          sub_2BFD020(a1, v7);
          v26 = *(_DWORD *)(a1 + 24);
          if ( !v26 )
          {
LABEL_48:
            ++*(_DWORD *)(a1 + 16);
            BUG();
          }
          v27 = v26 - 1;
          v28 = *(_QWORD *)(a1 + 8);
          v29 = 0;
          v30 = v27 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
          v5 = v37;
          v4 = v34;
          v31 = 1;
          v19 = *(_DWORD *)(a1 + 16) + 1;
          v20 = (_QWORD *)(v28 + 16LL * v30);
          v32 = *v20;
          if ( v8 != *v20 )
          {
            while ( v32 != -4096 )
            {
              if ( !v29 && v32 == -8192 )
                v29 = v20;
              v30 = v27 & (v31 + v30);
              v20 = (_QWORD *)(v28 + 16LL * v30);
              v32 = *v20;
              if ( v8 == *v20 )
                goto LABEL_24;
              ++v31;
            }
            if ( v29 )
              v20 = v29;
          }
        }
LABEL_24:
        *(_DWORD *)(a1 + 16) = v19;
        if ( *v20 != -4096 )
          --*(_DWORD *)(a1 + 20);
        *v20 = v8;
        v13 = v20 + 1;
        v20[1] = 0;
        goto LABEL_6;
      }
LABEL_5:
      v13 = v11 + 1;
LABEL_6:
      *v13 = v4;
    }
  }
  return v4;
}
