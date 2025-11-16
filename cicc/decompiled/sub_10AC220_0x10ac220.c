// Function: sub_10AC220
// Address: 0x10ac220
//
bool __fastcall sub_10AC220(__int64 a1, int a2, unsigned __int8 *a3)
{
  bool result; // al
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx

  if ( a2 + 29 != *a3 )
    return 0;
  v5 = *((_QWORD *)a3 - 8);
  if ( *(_BYTE *)v5 == 85
    && (v7 = *(_QWORD *)(v5 - 32)) != 0
    && !*(_BYTE *)v7
    && *(_QWORD *)(v7 + 24) == *(_QWORD *)(v5 + 80)
    && *(_DWORD *)(v7 + 36) == *(_DWORD *)a1
    && (v8 = *(_QWORD *)(v5 + 32 * (*(unsigned int *)(a1 + 8) - (unsigned __int64)(*(_DWORD *)(v5 + 4) & 0x7FFFFFF)))) != 0
    && (**(_QWORD **)(a1 + 16) = v8, *(_BYTE *)v5 == 85)
    && (v9 = *(_QWORD *)(v5 + 32 * (*(unsigned int *)(a1 + 24) - (unsigned __int64)(*(_DWORD *)(v5 + 4) & 0x7FFFFFF)))) != 0 )
  {
    **(_QWORD **)(a1 + 32) = v9;
    v6 = *((_QWORD *)a3 - 4);
    result = sub_10AC150(a1 + 40, v6);
    if ( result )
      return result;
  }
  else
  {
    v6 = *((_QWORD *)a3 - 4);
  }
  if ( *(_BYTE *)v6 != 85 )
    return 0;
  v10 = *(_QWORD *)(v6 - 32);
  if ( !v10 )
    return 0;
  if ( *(_BYTE *)v10 )
    return 0;
  if ( *(_QWORD *)(v10 + 24) != *(_QWORD *)(v6 + 80) )
    return 0;
  if ( *(_DWORD *)(v10 + 36) != *(_DWORD *)a1 )
    return 0;
  v11 = *(_QWORD *)(v6 + 32 * (*(unsigned int *)(a1 + 8) - (unsigned __int64)(*(_DWORD *)(v6 + 4) & 0x7FFFFFF)));
  if ( !v11 )
    return 0;
  **(_QWORD **)(a1 + 16) = v11;
  if ( *(_BYTE *)v6 != 85 )
    return 0;
  v12 = *(_QWORD *)(v6 + 32 * (*(unsigned int *)(a1 + 24) - (unsigned __int64)(*(_DWORD *)(v6 + 4) & 0x7FFFFFF)));
  if ( !v12 )
    return 0;
  **(_QWORD **)(a1 + 32) = v12;
  return sub_10AC150(a1 + 40, *((_QWORD *)a3 - 8));
}
