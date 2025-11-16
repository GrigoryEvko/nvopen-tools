// Function: sub_266F260
// Address: 0x266f260
//
__int64 __fastcall sub_266F260(__int64 a1, __int64 a2)
{
  __int64 v3; // rdx
  size_t v4; // rdx
  __int64 v5; // rdx
  size_t v6; // rdx
  __int64 v7; // rdx
  size_t v8; // rdx
  __int64 v9; // rdx
  size_t v10; // rdx
  unsigned int v11; // r12d
  size_t v12; // rdx

  if ( *(_BYTE *)(a1 + 153) != *(_BYTE *)(a2 + 153) )
    return 0;
  if ( *(_BYTE *)(a1 + 152) != *(_BYTE *)(a2 + 152) )
    return 0;
  v3 = *(unsigned int *)(a1 + 200);
  if ( v3 != *(_DWORD *)(a2 + 200) )
    return 0;
  v4 = 8 * v3;
  if ( v4 )
  {
    if ( memcmp(*(const void **)(a1 + 192), *(const void **)(a2 + 192), v4) )
      return 0;
  }
  if ( *(_BYTE *)(a1 + 25) != *(_BYTE *)(a2 + 25) )
    return 0;
  if ( *(_BYTE *)(a1 + 24) != *(_BYTE *)(a2 + 24) )
    return 0;
  v5 = *(unsigned int *)(a1 + 72);
  if ( v5 != *(_DWORD *)(a2 + 72) )
    return 0;
  v6 = 8 * v5;
  if ( v6 )
  {
    if ( memcmp(*(const void **)(a1 + 64), *(const void **)(a2 + 64), v6) )
      return 0;
  }
  if ( *(_BYTE *)(a1 + 89) != *(_BYTE *)(a2 + 89) )
    return 0;
  if ( *(_BYTE *)(a1 + 88) != *(_BYTE *)(a2 + 88) )
    return 0;
  v7 = *(unsigned int *)(a1 + 136);
  if ( v7 != *(_DWORD *)(a2 + 136) )
    return 0;
  v8 = 8 * v7;
  if ( v8 )
  {
    if ( memcmp(*(const void **)(a1 + 128), *(const void **)(a2 + 128), v8) )
      return 0;
  }
  if ( *(_BYTE *)(a1 + 249) != *(_BYTE *)(a2 + 249) )
    return 0;
  if ( *(_BYTE *)(a1 + 248) != *(_BYTE *)(a2 + 248) )
    return 0;
  v9 = *(unsigned int *)(a1 + 296);
  if ( v9 != *(_DWORD *)(a2 + 296) )
    return 0;
  v10 = 8 * v9;
  if ( v10 )
  {
    if ( memcmp(*(const void **)(a1 + 288), *(const void **)(a2 + 288), v10) )
      return 0;
  }
  v11 = 0;
  if ( *(_BYTE *)(a1 + 313) != *(_BYTE *)(a2 + 313) )
    return 0;
  if ( *(_BYTE *)(a1 + 312) != *(_BYTE *)(a2 + 312) )
    return 0;
  v12 = *(_QWORD *)(a1 + 360);
  if ( v12 != *(_QWORD *)(a2 + 360) || v12 && memcmp(*(const void **)(a1 + 352), *(const void **)(a2 + 352), v12) )
    return 0;
  LOBYTE(v11) = *(_BYTE *)(a1 + 376) == *(_BYTE *)(a2 + 376);
  return v11;
}
