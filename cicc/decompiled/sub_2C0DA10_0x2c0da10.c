// Function: sub_2C0DA10
// Address: 0x2c0da10
//
bool __fastcall sub_2C0DA10(__int64 a1)
{
  __int64 **v1; // rbx

  v1 = (__int64 **)(*(_QWORD *)(a1 + 72) + 8LL * *(unsigned int *)(a1 + 80));
  return v1 == sub_2C0D840(*(__int64 ***)(a1 + 72), (__int64)v1, a1 - 40);
}
