// Function: sub_2E32F70
// Address: 0x2e32f70
//
__int64 __fastcall sub_2E32F70(__int64 a1, __int64 a2)
{
  return *(_QWORD *)(a1 + 144) + 4 * ((a2 - *(_QWORD *)(a1 + 112)) >> 3);
}
