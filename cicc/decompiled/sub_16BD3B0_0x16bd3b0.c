// Function: sub_16BD3B0
// Address: 0x16bd3b0
//
bool __fastcall sub_16BD3B0(__int64 a1, const void *a2, __int64 a3)
{
  __int64 v4; // rdx

  v4 = *(_QWORD *)(a1 + 8);
  return v4 == a3 && memcmp(*(const void **)a1, a2, 4 * v4) == 0;
}
