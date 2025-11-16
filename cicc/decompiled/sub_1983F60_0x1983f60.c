// Function: sub_1983F60
// Address: 0x1983f60
//
_BOOL8 __fastcall sub_1983F60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v5; // r8
  int v6; // r9d
  unsigned int v7; // edi
  __int64 *v8; // rdx
  __int64 v9; // r10
  __int64 *v10; // rax
  unsigned int v11; // edi
  __int64 *v12; // rsi
  __int64 v13; // r10
  int v15; // edx
  int v16; // r11d
  int v17; // esi
  int v18; // r11d

  v3 = *(unsigned int *)(a1 + 2472);
  if ( !(_DWORD)v3 )
    return 0;
  v5 = *(_QWORD *)(a1 + 2456);
  v6 = v3 - 1;
  v7 = (v3 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v15 = 1;
    while ( v9 != -8 )
    {
      v16 = v15 + 1;
      v7 = v6 & (v15 + v7);
      v8 = (__int64 *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      v15 = v16;
    }
    return 0;
  }
LABEL_3:
  v10 = (__int64 *)(v5 + 16 * v3);
  if ( v8 == v10 )
    return 0;
  v11 = v6 & (((unsigned int)a3 >> 9) ^ ((unsigned int)a3 >> 4));
  v12 = (__int64 *)(v5 + 16LL * v11);
  v13 = *v12;
  if ( a3 != *v12 )
  {
    v17 = 1;
    while ( v13 != -8 )
    {
      v18 = v17 + 1;
      v11 = v6 & (v17 + v11);
      v12 = (__int64 *)(v5 + 16LL * v11);
      v13 = *v12;
      if ( a3 == *v12 )
        return v12 != v10 && *((_DWORD *)v8 + 2) == *((_DWORD *)v12 + 2);
      v17 = v18;
    }
    return 0;
  }
  return v12 != v10 && *((_DWORD *)v8 + 2) == *((_DWORD *)v12 + 2);
}
