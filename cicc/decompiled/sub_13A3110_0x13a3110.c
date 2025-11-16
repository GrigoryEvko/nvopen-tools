// Function: sub_13A3110
// Address: 0x13a3110
//
__int64 __fastcall sub_13A3110(__int64 a1, int a2)
{
  return *(_BYTE *)(*(_QWORD *)(a1 + 48) + 16LL * (unsigned int)(a2 - 1)) & 7;
}
