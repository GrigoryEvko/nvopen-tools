// Function: sub_F8E510
// Address: 0xf8e510
//
__int64 __fastcall sub_F8E510(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __int64 v4; // rax
  __int64 v5; // rax

  v2 = a1;
  if ( *(_BYTE *)a1 == 17 )
    return v2;
  if ( *(_BYTE *)a1 > 0x15u )
    return 0;
  v4 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v4 + 8) != 14 || *((_BYTE *)sub_AE2980(a2, *(_DWORD *)(v4 + 8) >> 8) + 16) )
    return 0;
  v5 = sub_AE4450(a2, *(_QWORD *)(a1 + 8));
  if ( *(_BYTE *)a1 == 20 )
    return sub_ACD640(v5, 0, 0);
  if ( *(_BYTE *)a1 != 5 )
    return 0;
  if ( *(_WORD *)(a1 + 2) != 48 )
    return 0;
  v2 = *(_QWORD *)(a1 - 32LL * (*(_DWORD *)(a1 + 4) & 0x7FFFFFF));
  if ( *(_BYTE *)v2 != 17 )
    return 0;
  if ( v5 == *(_QWORD *)(v2 + 8) )
    return v2;
  return sub_96F3F0(v2, v5, 0, a2);
}
