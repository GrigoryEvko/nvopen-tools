// Function: sub_2BFDEB0
// Address: 0x2bfdeb0
//
__int64 __fastcall sub_2BFDEB0(__int64 *a1)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 v4; // r8
  unsigned int v5; // r12d
  __int64 v6; // r9
  int v7; // r11d
  _QWORD *v8; // rdx
  unsigned int v9; // edi
  _QWORD *v10; // rax
  __int64 v11; // rcx
  __int64 *v12; // rax
  __int64 v13; // rbx
  unsigned int v14; // esi
  __int64 v15; // r13
  int v16; // eax
  int v17; // esi
  __int64 v18; // rdi
  unsigned int v19; // eax
  int v20; // ecx
  __int64 v21; // r9
  int v22; // eax
  int v23; // eax
  int v24; // eax
  __int64 v25; // rdi
  _QWORD *v26; // r9
  unsigned int v27; // r15d
  int v28; // r10d
  __int64 v29; // rsi
  int v31; // r11d
  _QWORD *v32; // r10
  __int64 v33; // [rsp+8h] [rbp-38h]
  __int64 v34; // [rsp+8h] [rbp-38h]

  v2 = sub_2BFD6A0(*a1, **(_QWORD **)(a1[1] + 48));
  v3 = a1[1];
  v4 = v2;
  if ( *(_DWORD *)(v3 + 56) != 1 )
  {
    v5 = 1;
    while ( 1 )
    {
      v13 = *a1;
      v14 = *(_DWORD *)(*a1 + 24);
      v15 = *(_QWORD *)(*(_QWORD *)(v3 + 48) + 8LL * v5);
      if ( !v14 )
        break;
      v6 = *(_QWORD *)(v13 + 8);
      v7 = 1;
      v8 = 0;
      v9 = (v14 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
      v10 = (_QWORD *)(v6 + 16LL * v9);
      v11 = *v10;
      if ( v15 != *v10 )
      {
        while ( v11 != -4096 )
        {
          if ( !v8 && v11 == -8192 )
            v8 = v10;
          v9 = (v14 - 1) & (v7 + v9);
          v10 = (_QWORD *)(v6 + 16LL * v9);
          v11 = *v10;
          if ( v15 == *v10 )
            goto LABEL_4;
          ++v7;
        }
        if ( !v8 )
          v8 = v10;
        v22 = *(_DWORD *)(v13 + 16);
        ++*(_QWORD *)v13;
        v20 = v22 + 1;
        if ( 4 * (v22 + 1) < 3 * v14 )
        {
          if ( v14 - *(_DWORD *)(v13 + 20) - v20 <= v14 >> 3 )
          {
            v34 = v4;
            sub_2BFD020(v13, v14);
            v23 = *(_DWORD *)(v13 + 24);
            if ( !v23 )
            {
LABEL_45:
              ++*(_DWORD *)(v13 + 16);
              BUG();
            }
            v24 = v23 - 1;
            v25 = *(_QWORD *)(v13 + 8);
            v26 = 0;
            v27 = v24 & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
            v4 = v34;
            v28 = 1;
            v20 = *(_DWORD *)(v13 + 16) + 1;
            v8 = (_QWORD *)(v25 + 16LL * v27);
            v29 = *v8;
            if ( v15 != *v8 )
            {
              while ( v29 != -4096 )
              {
                if ( !v26 && v29 == -8192 )
                  v26 = v8;
                v27 = v24 & (v28 + v27);
                v8 = (_QWORD *)(v25 + 16LL * v27);
                v29 = *v8;
                if ( v15 == *v8 )
                  goto LABEL_10;
                ++v28;
              }
              if ( v26 )
                v8 = v26;
            }
          }
          goto LABEL_10;
        }
LABEL_8:
        v33 = v4;
        sub_2BFD020(v13, 2 * v14);
        v16 = *(_DWORD *)(v13 + 24);
        if ( !v16 )
          goto LABEL_45;
        v17 = v16 - 1;
        v18 = *(_QWORD *)(v13 + 8);
        v4 = v33;
        v19 = (v16 - 1) & (((unsigned int)v15 >> 9) ^ ((unsigned int)v15 >> 4));
        v20 = *(_DWORD *)(v13 + 16) + 1;
        v8 = (_QWORD *)(v18 + 16LL * v19);
        v21 = *v8;
        if ( v15 != *v8 )
        {
          v31 = 1;
          v32 = 0;
          while ( v21 != -4096 )
          {
            if ( !v32 && v21 == -8192 )
              v32 = v8;
            v19 = v17 & (v31 + v19);
            v8 = (_QWORD *)(v18 + 16LL * v19);
            v21 = *v8;
            if ( v15 == *v8 )
              goto LABEL_10;
            ++v31;
          }
          if ( v32 )
            v8 = v32;
        }
LABEL_10:
        *(_DWORD *)(v13 + 16) = v20;
        if ( *v8 != -4096 )
          --*(_DWORD *)(v13 + 20);
        *v8 = v15;
        v12 = v8 + 1;
        v8[1] = 0;
        goto LABEL_5;
      }
LABEL_4:
      v12 = v10 + 1;
LABEL_5:
      *v12 = v4;
      v3 = a1[1];
      if ( ++v5 == *(_DWORD *)(v3 + 56) )
        return v4;
    }
    ++*(_QWORD *)v13;
    goto LABEL_8;
  }
  return v4;
}
