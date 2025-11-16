// Function: sub_2E1D650
// Address: 0x2e1d650
//
bool __fastcall sub_2E1D650(__int64 *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 *v7; // r9
  __int64 v8; // rsi
  __int64 v9; // rax
  unsigned __int64 v10; // r8
  unsigned int v11; // edx
  __int64 v12; // rcx
  __int64 *v13; // rsi
  unsigned int v14; // eax
  bool result; // al
  unsigned int v16; // eax
  unsigned int v17; // eax
  unsigned int v18; // eax
  unsigned int v19; // edx
  unsigned int v20; // ecx
  unsigned int v21; // eax
  unsigned int v22; // eax

  v6 = 8 * a2;
  v7 = &a1[a2];
  v8 = (8 * a2) >> 5;
  v9 = v6 >> 3;
  if ( v8 > 0 )
  {
    v10 = a4 & 0xFFFFFFFFFFFFFFF8LL;
    v11 = *(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a3 >> 1) & 3;
    v12 = (a4 >> 1) & 3;
    v13 = &a1[4 * v8];
    while ( 1 )
    {
      v14 = *(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a1 >> 1) & 3;
      if ( v11 <= v14 && v14 < ((unsigned int)v12 | *(_DWORD *)(v10 + 24)) )
        return v7 != a1;
      v16 = *(_DWORD *)((a1[1] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a1[1] >> 1) & 3;
      if ( v11 <= v16 && v16 < ((unsigned int)v12 | *(_DWORD *)(v10 + 24)) )
        return v7 != a1 + 1;
      v17 = *(_DWORD *)((a1[2] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a1[2] >> 1) & 3;
      if ( v11 <= v17 && v17 < ((unsigned int)v12 | *(_DWORD *)(v10 + 24)) )
        return v7 != a1 + 2;
      v18 = *(_DWORD *)((a1[3] & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a1[3] >> 1) & 3;
      if ( v11 <= v18 && v18 < ((unsigned int)v12 | *(_DWORD *)(v10 + 24)) )
        return v7 != a1 + 3;
      a1 += 4;
      if ( v13 == a1 )
      {
        v9 = v7 - a1;
        break;
      }
    }
  }
  switch ( v9 )
  {
    case 2LL:
      v19 = *(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a3 >> 1) & 3;
      break;
    case 3LL:
      v19 = *(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a3 >> 1) & 3;
      v21 = *(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a1 >> 1) & 3;
      if ( v19 <= v21 && v21 < (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a4 >> 1) & 3) )
        return v7 != a1;
      ++a1;
      break;
    case 1LL:
      v19 = *(_DWORD *)((a3 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (a3 >> 1) & 3;
      goto LABEL_23;
    default:
      return 0;
  }
  v22 = *(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a1 >> 1) & 3;
  if ( v19 <= v22 && v22 < (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a4 >> 1) & 3) )
    return v7 != a1;
  ++a1;
LABEL_23:
  result = 0;
  v20 = *(_DWORD *)((*a1 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (*a1 >> 1) & 3;
  if ( v19 <= v20 && v20 < (*(_DWORD *)((a4 & 0xFFFFFFFFFFFFFFF8LL) + 24) | (unsigned int)(a4 >> 1) & 3) )
    return v7 != a1;
  return result;
}
