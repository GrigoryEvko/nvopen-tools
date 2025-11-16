// Function: sub_15A8140
// Address: 0x15a8140
//
__int64 __fastcall sub_15A8140(__int64 a1, __int64 a2)
{
  size_t v3; // rdx
  __int64 v4; // rax
  _DWORD *v5; // r13
  _DWORD *v6; // r14
  _DWORD *v7; // r15
  __int64 v8; // rax
  _DWORD *v9; // rbx
  _DWORD *v10; // r12
  _DWORD *v11; // r13

  if ( (*(_QWORD *)a1 & 0xFFFFFFFF000000FFLL) != (*(_QWORD *)a2 & 0xFFFFFFFF000000FFLL) )
    return 0;
  if ( *(_QWORD *)(a1 + 8) != *(_QWORD *)(a2 + 8) )
    return 0;
  if ( *(_DWORD *)(a1 + 16) != *(_DWORD *)(a2 + 16) )
    return 0;
  v3 = *(unsigned int *)(a1 + 32);
  if ( v3 != *(_DWORD *)(a2 + 32)
    || *(_DWORD *)(a1 + 32) && memcmp(*(const void **)(a1 + 24), *(const void **)(a2 + 24), v3) )
  {
    return 0;
  }
  v4 = *(unsigned int *)(a1 + 56);
  if ( v4 != *(_DWORD *)(a2 + 56) )
    return 0;
  v5 = *(_DWORD **)(a1 + 48);
  v6 = *(_DWORD **)(a2 + 48);
  v7 = &v5[2 * v4];
  if ( v5 != v7 )
  {
    while ( sub_15A80B0(v5, v6) )
    {
      v5 += 2;
      v6 += 2;
      if ( v7 == v5 )
        goto LABEL_15;
    }
    return 0;
  }
LABEL_15:
  v8 = *(unsigned int *)(a1 + 232);
  if ( v8 != *(_DWORD *)(a2 + 232) )
    return 0;
  v9 = *(_DWORD **)(a1 + 224);
  v10 = *(_DWORD **)(a2 + 224);
  v11 = &v9[5 * v8];
  if ( v9 != v11 )
  {
    while ( sub_15A8100(v9, v10) )
    {
      v9 += 5;
      v10 += 5;
      if ( v11 == v9 )
        return 1;
    }
    return 0;
  }
  return 1;
}
