// Function: sub_26EDB30
// Address: 0x26edb30
//
void __fastcall sub_26EDB30(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 v3; // r14
  __int64 i; // rax
  __int64 v7; // r12
  __int64 v8; // rdx
  char v9; // cl
  __int64 v10; // rax
  int v11; // ecx
  unsigned int v12; // esi
  __int64 v13; // r10
  unsigned int v14; // edi
  _QWORD *v15; // rax
  __int64 v16; // rcx
  int v17; // r9d
  _QWORD *v18; // r8
  int v19; // edi
  int v20; // ecx
  int v21; // r8d
  int v22; // r8d
  __int64 v23; // r9
  unsigned int v24; // edx
  __int64 v25; // r11
  int v26; // edi
  _QWORD *v27; // rsi
  int v28; // r9d
  int v29; // r9d
  __int64 v30; // r10
  _QWORD *v31; // r8
  __int64 v32; // rdx
  int v33; // esi
  __int64 v34; // rdi
  unsigned int v35; // [rsp+Ch] [rbp-34h]

  v2 = *(_QWORD *)(*(_QWORD *)a1 + 80LL);
  v3 = *(_QWORD *)a1 + 72LL;
  if ( v2 != v3 )
  {
    for ( i = *(_QWORD *)(*(_QWORD *)a1 + 80LL); ; i = *(_QWORD *)(*(_QWORD *)a1 + 80LL) )
    {
      if ( !v2 )
      {
        if ( i )
          BUG();
LABEL_3:
        v2 = *(_QWORD *)(v2 + 8);
        if ( v3 == v2 )
          return;
        continue;
      }
      v7 = v2 - 24;
      if ( i && v7 == i - 24 )
        goto LABEL_3;
      v8 = *(_QWORD *)(v2 - 8);
      do
      {
        if ( !v8 )
          goto LABEL_16;
        v9 = **(_BYTE **)(v8 + 24);
        v10 = v8;
        v8 = *(_QWORD *)(v8 + 8);
      }
      while ( (unsigned __int8)(v9 - 30) > 0xAu );
      v11 = 0;
      while ( 1 )
      {
        v10 = *(_QWORD *)(v10 + 8);
        if ( !v10 )
          break;
        while ( (unsigned __int8)(**(_BYTE **)(v10 + 24) - 30) <= 0xAu )
        {
          v10 = *(_QWORD *)(v10 + 8);
          ++v11;
          if ( !v10 )
            goto LABEL_15;
        }
      }
LABEL_15:
      if ( v11 != -1 )
        goto LABEL_3;
LABEL_16:
      v12 = *(_DWORD *)(a2 + 24);
      if ( !v12 )
        break;
      v13 = *(_QWORD *)(a2 + 8);
      v14 = (v12 - 1) & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
      v15 = (_QWORD *)(v13 + 8LL * v14);
      v16 = *v15;
      if ( v7 == *v15 )
        goto LABEL_3;
      v17 = 1;
      v18 = 0;
      while ( v16 != -4096 )
      {
        if ( v16 == -8192 && !v18 )
          v18 = v15;
        v14 = (v12 - 1) & (v17 + v14);
        v15 = (_QWORD *)(v13 + 8LL * v14);
        v16 = *v15;
        if ( v7 == *v15 )
          goto LABEL_3;
        ++v17;
      }
      v19 = *(_DWORD *)(a2 + 16);
      if ( v18 )
        v15 = v18;
      ++*(_QWORD *)a2;
      v20 = v19 + 1;
      if ( 4 * (v19 + 1) >= 3 * v12 )
        goto LABEL_31;
      if ( v12 - *(_DWORD *)(a2 + 20) - v20 <= v12 >> 3 )
      {
        v35 = ((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4);
        sub_CF28B0(a2, v12);
        v28 = *(_DWORD *)(a2 + 24);
        if ( !v28 )
        {
LABEL_59:
          ++*(_DWORD *)(a2 + 16);
          BUG();
        }
        v29 = v28 - 1;
        v30 = *(_QWORD *)(a2 + 8);
        v31 = 0;
        LODWORD(v32) = v29 & v35;
        v20 = *(_DWORD *)(a2 + 16) + 1;
        v33 = 1;
        v15 = (_QWORD *)(v30 + 8LL * (v29 & v35));
        v34 = *v15;
        if ( v7 != *v15 )
        {
          while ( v34 != -4096 )
          {
            if ( !v31 && v34 == -8192 )
              v31 = v15;
            v32 = v29 & (unsigned int)(v32 + v33);
            v15 = (_QWORD *)(v30 + 8 * v32);
            v34 = *v15;
            if ( v7 == *v15 )
              goto LABEL_24;
            ++v33;
          }
          if ( v31 )
            v15 = v31;
        }
      }
LABEL_24:
      *(_DWORD *)(a2 + 16) = v20;
      if ( *v15 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v15 = v7;
      v2 = *(_QWORD *)(v2 + 8);
      if ( v3 == v2 )
        return;
    }
    ++*(_QWORD *)a2;
LABEL_31:
    sub_CF28B0(a2, 2 * v12);
    v21 = *(_DWORD *)(a2 + 24);
    if ( !v21 )
      goto LABEL_59;
    v22 = v21 - 1;
    v23 = *(_QWORD *)(a2 + 8);
    v24 = v22 & (((unsigned int)v7 >> 9) ^ ((unsigned int)v7 >> 4));
    v20 = *(_DWORD *)(a2 + 16) + 1;
    v15 = (_QWORD *)(v23 + 8LL * v24);
    v25 = *v15;
    if ( v7 != *v15 )
    {
      v26 = 1;
      v27 = 0;
      while ( v25 != -4096 )
      {
        if ( !v27 && v25 == -8192 )
          v27 = v15;
        v24 = v22 & (v26 + v24);
        v15 = (_QWORD *)(v23 + 8LL * v24);
        v25 = *v15;
        if ( v7 == *v15 )
          goto LABEL_24;
        ++v26;
      }
      if ( v27 )
        v15 = v27;
    }
    goto LABEL_24;
  }
}
