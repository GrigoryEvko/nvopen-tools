// Function: sub_B32750
// Address: 0xb32750
//
__int64 __fastcall sub_B32750(__int64 a1, __int64 a2, __int64 a3)
{
  char v6; // al
  __int64 v7; // r13
  unsigned int v9; // esi
  __int64 v10; // rdi
  int v11; // r10d
  __int64 *v12; // r9
  unsigned int v13; // ecx
  __int64 *v14; // rdx
  __int64 v15; // rax
  unsigned __int16 v16; // ax
  __int64 v17; // rax
  int v18; // eax
  int v19; // eax
  int v20; // eax
  int v21; // ecx
  __int64 v22; // rdi
  unsigned int v23; // edx
  __int64 v24; // rsi
  int v25; // r10d
  __int64 *v26; // r8
  int v27; // eax
  int v28; // edx
  __int64 v29; // rsi
  int v30; // r8d
  __int64 *v31; // rdi
  unsigned int v32; // r13d
  __int64 v33; // rcx

  while ( 1 )
  {
    while ( 1 )
    {
      v6 = *(_BYTE *)a1;
      if ( (unsigned __int8)(*(_BYTE *)a1 - 2) <= 1u || !v6 )
      {
        v7 = a1;
        (*(void (__fastcall **)(_QWORD, __int64))a3)(*(_QWORD *)(a3 + 8), a1);
        return v7;
      }
      if ( v6 != 1 )
        break;
      (*(void (__fastcall **)(_QWORD, __int64))a3)(*(_QWORD *)(a3 + 8), a1);
      v9 = *(_DWORD *)(a2 + 24);
      if ( !v9 )
      {
        ++*(_QWORD *)a2;
LABEL_36:
        sub_B32090(a2, 2 * v9);
        v20 = *(_DWORD *)(a2 + 24);
        if ( !v20 )
          goto LABEL_59;
        v21 = v20 - 1;
        v22 = *(_QWORD *)(a2 + 8);
        v23 = (v20 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v12 = (__int64 *)(v22 + 8LL * v23);
        v24 = *v12;
        v19 = *(_DWORD *)(a2 + 16) + 1;
        if ( *v12 != a1 )
        {
          v25 = 1;
          v26 = 0;
          while ( v24 != -4096 )
          {
            if ( v24 == -8192 && !v26 )
              v26 = v12;
            v23 = v21 & (v25 + v23);
            v12 = (__int64 *)(v22 + 8LL * v23);
            v24 = *v12;
            if ( *v12 == a1 )
              goto LABEL_32;
            ++v25;
          }
          if ( v26 )
            v12 = v26;
        }
        goto LABEL_32;
      }
      v10 = *(_QWORD *)(a2 + 8);
      v11 = 1;
      v12 = 0;
      v13 = (v9 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
      v14 = (__int64 *)(v10 + 8LL * v13);
      v15 = *v14;
      if ( *v14 == a1 )
        break;
      while ( v15 != -4096 )
      {
        if ( !v12 && v15 == -8192 )
          v12 = v14;
        v13 = (v9 - 1) & (v11 + v13);
        v14 = (__int64 *)(v10 + 8LL * v13);
        v15 = *v14;
        if ( *v14 == a1 )
          goto LABEL_8;
        ++v11;
      }
      v18 = *(_DWORD *)(a2 + 16);
      if ( !v12 )
        v12 = v14;
      ++*(_QWORD *)a2;
      v19 = v18 + 1;
      if ( 4 * v19 >= 3 * v9 )
        goto LABEL_36;
      if ( v9 - (v19 + *(_DWORD *)(a2 + 20)) <= v9 >> 3 )
      {
        sub_B32090(a2, v9);
        v27 = *(_DWORD *)(a2 + 24);
        if ( !v27 )
        {
LABEL_59:
          ++*(_DWORD *)(a2 + 16);
          BUG();
        }
        v28 = v27 - 1;
        v29 = *(_QWORD *)(a2 + 8);
        v30 = 1;
        v31 = 0;
        v32 = (v27 - 1) & (((unsigned int)a1 >> 9) ^ ((unsigned int)a1 >> 4));
        v12 = (__int64 *)(v29 + 8LL * v32);
        v33 = *v12;
        v19 = *(_DWORD *)(a2 + 16) + 1;
        if ( *v12 != a1 )
        {
          while ( v33 != -4096 )
          {
            if ( !v31 && v33 == -8192 )
              v31 = v12;
            v32 = v28 & (v30 + v32);
            v12 = (__int64 *)(v29 + 8LL * v32);
            v33 = *v12;
            if ( *v12 == a1 )
              goto LABEL_32;
            ++v30;
          }
          if ( v31 )
            v12 = v31;
        }
      }
LABEL_32:
      *(_DWORD *)(a2 + 16) = v19;
      if ( *v12 != -4096 )
        --*(_DWORD *)(a2 + 20);
      *v12 = a1;
      a1 = *(_QWORD *)(a1 - 32);
    }
LABEL_8:
    if ( *(_BYTE *)a1 != 5 )
      return 0;
    v16 = *(_WORD *)(a1 + 2);
    if ( v16 == 15 )
    {
      if ( sub_B32750(*(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))), a2, a3) )
        return 0;
      goto LABEL_21;
    }
    if ( v16 <= 0xFu )
      break;
    if ( v16 != 34 && (unsigned __int16)(v16 - 47) > 2u )
      return 0;
LABEL_21:
    a1 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  }
  if ( v16 == 13 )
  {
    v7 = sub_B32750(*(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)), a2, a3);
    v17 = sub_B32750(*(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))), a2, a3);
    if ( !v7 || !v17 )
    {
      if ( !v7 )
        return v17;
      return v7;
    }
  }
  return 0;
}
