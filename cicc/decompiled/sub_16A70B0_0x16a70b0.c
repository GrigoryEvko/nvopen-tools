// Function: sub_16A70B0
// Address: 0x16a70b0
//
_BOOL8 __fastcall sub_16A70B0(__int64 a1, unsigned int a2)
{
  return (*(_QWORD *)(a1 + 8LL * (a2 >> 6)) & (1LL << a2)) != 0;
}
