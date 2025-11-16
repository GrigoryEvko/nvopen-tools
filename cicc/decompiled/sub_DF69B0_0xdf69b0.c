// Function: sub_DF69B0
// Address: 0xdf69b0
//
bool __fastcall sub_DF69B0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rbx
  __int64 v5; // r8
  bool result; // al
  __int64 v7; // rbx

  v4 = sub_B2D7E0(a3, "target-cpu", 0xAu);
  v5 = sub_B2D7E0(a2, "target-cpu", 0xAu);
  result = 0;
  if ( v4 == v5 )
  {
    v7 = sub_B2D7E0(a3, "target-features", 0xFu);
    return v7 == sub_B2D7E0(a2, "target-features", 0xFu);
  }
  return result;
}
