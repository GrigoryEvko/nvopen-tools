// Function: sub_16F7930
// Address: 0x16f7930
//
__int64 __fastcall sub_16F7930(__int64 a1, unsigned int a2)
{
  *(_DWORD *)(a1 + 60) += a2;
  *(_QWORD *)(a1 + 40) += a2;
  return a2;
}
