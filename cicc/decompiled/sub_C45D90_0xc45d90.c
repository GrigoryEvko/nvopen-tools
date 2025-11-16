// Function: sub_C45D90
// Address: 0xc45d90
//
_BOOL8 __fastcall sub_C45D90(__int64 a1, unsigned int a2)
{
  return (*(_QWORD *)(a1 + 8LL * (a2 >> 6)) & (1LL << a2)) != 0;
}
