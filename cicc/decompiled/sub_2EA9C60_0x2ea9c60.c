// Function: sub_2EA9C60
// Address: 0x2ea9c60
//
__int64 __fastcall sub_2EA9C60(__int64 a1, __int64 *a2, _QWORD *a3)
{
  unsigned int v4; // eax
  __int64 result; // rax
  __int64 v6; // rdx
  __int64 v7; // rax
  unsigned int v8; // esi
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rdx

  v4 = sub_C8ED90(a2, *(_QWORD *)(a1 + 8));
  if ( !v4 )
    return 0;
  if ( v4 > (unsigned __int64)((__int64)(a3[1] - *a3) >> 3) )
    return 0;
  v6 = *(_QWORD *)(*a3 + 8LL * (v4 - 1));
  if ( !v6 )
    return 0;
  v7 = (unsigned int)(*(_DWORD *)(a1 + 48) - 1);
  if ( (*(_BYTE *)(v6 - 16) & 2) != 0 )
    v8 = *(_DWORD *)(v6 - 24);
  else
    v8 = (*(_WORD *)(v6 - 16) >> 6) & 0xF;
  if ( (unsigned int)v7 >= v8 )
    v7 = 0;
  if ( !v8 )
    return 0;
  v9 = (*(_BYTE *)(v6 - 16) & 2) != 0 ? *(_QWORD *)(v6 - 32) : v6 - 8LL * ((*(_BYTE *)(v6 - 16) >> 2) & 0xF) - 16;
  v10 = *(_QWORD *)(v9 + 8 * v7);
  if ( *(_BYTE *)v10 != 1 )
    return 0;
  v11 = *(_QWORD *)(v10 + 136);
  if ( *(_BYTE *)v11 != 17 )
    return 0;
  result = *(_QWORD *)(v11 + 24);
  if ( *(_DWORD *)(v11 + 32) > 0x40u )
    return *(_QWORD *)result;
  return result;
}
