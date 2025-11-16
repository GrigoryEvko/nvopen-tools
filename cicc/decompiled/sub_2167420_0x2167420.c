// Function: sub_2167420
// Address: 0x2167420
//
__int64 __fastcall sub_2167420(__int64 a1, unsigned int a2, __int64 a3)
{
  unsigned __int8 v3; // al
  unsigned int v4; // r8d

  v3 = sub_2167220(*(_QWORD *)(a1 + 8), a3);
  v4 = 0;
  if ( v3 )
    LOBYTE(v4) = (*(_BYTE *)(a2 + *(_QWORD *)(a1 + 24) + 5LL * v3 + 71883) & 0xB) == 0;
  return v4;
}
