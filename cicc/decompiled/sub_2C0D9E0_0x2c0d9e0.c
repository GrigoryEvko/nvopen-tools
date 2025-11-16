// Function: sub_2C0D9E0
// Address: 0x2c0d9e0
//
bool __fastcall sub_2C0D9E0(__int64 a1)
{
  __int64 **v2; // rdi
  __int64 **v3; // rbx

  v2 = *(__int64 ***)(a1 + 112);
  v3 = &v2[*(unsigned int *)(a1 + 120)];
  return v3 == sub_2C0D840(v2, (__int64)v3, a1);
}
