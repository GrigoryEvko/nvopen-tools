// Function: sub_2E89060
// Address: 0x2e89060
//
__int64 __fastcall sub_2E89060(__int64 a1)
{
  __int64 result; // rax

  result = *(_QWORD *)(a1 + 8);
  *(_DWORD *)(a1 + 44) &= ~8u;
  *(_DWORD *)(result + 44) &= ~4u;
  return result;
}
