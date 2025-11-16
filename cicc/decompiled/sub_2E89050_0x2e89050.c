// Function: sub_2E89050
// Address: 0x2e89050
//
unsigned __int64 __fastcall sub_2E89050(__int64 *a1)
{
  __int64 v1; // rax
  unsigned __int64 result; // rax

  v1 = *a1;
  *((_DWORD *)a1 + 11) &= ~4u;
  result = v1 & 0xFFFFFFFFFFFFFFF8LL;
  *(_DWORD *)(result + 44) &= ~8u;
  return result;
}
