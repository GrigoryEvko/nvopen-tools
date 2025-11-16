// Function: sub_39B2070
// Address: 0x39b2070
//
__int64 __fastcall sub_39B2070(__int64 a1, unsigned int a2, __int64 a3)
{
  unsigned __int8 v3; // al
  unsigned int v4; // r8d

  v3 = sub_39B1E70(*(_QWORD *)(a1 + 8), a3);
  v4 = 0;
  if ( v3 )
    LOBYTE(v4) = (*(_BYTE *)(a2 + *(_QWORD *)(a1 + 24) + 5LL * v3 + 71883) & 0xB) == 0;
  return v4;
}
