// Function: sub_39F1E60
// Address: 0x39f1e60
//
__int64 __fastcall sub_39F1E60(__int64 a1, __int64 a2, __int64 a3, unsigned int a4)
{
  unsigned int v6; // ecx
  unsigned int v7; // ecx
  __int64 result; // rax

  sub_390D5F0(*(_QWORD *)(a1 + 264), a2, 0);
  *(_WORD *)(a2 + 8) |= 0xC10u;
  LOBYTE(v6) = 0;
  *(_QWORD *)(a2 + 24) = a3;
  if ( a4 )
  {
    _BitScanReverse(&v7, a4);
    v6 = -(v7 ^ 0x1F) & 0x1F;
  }
  result = *(_DWORD *)(a2 + 8) & 0xFFFE0FFF;
  *(_DWORD *)(a2 + 8) = result | ((v6 & 0x1F) << 12);
  return result;
}
