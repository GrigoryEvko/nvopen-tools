// Function: sub_14A0E50
// Address: 0x14a0e50
//
bool __fastcall sub_14A0E50(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // r8
  bool result; // al
  __int64 v7; // rbx

  v3 = a3 + 112;
  v4 = sub_1560340(a3 + 112, 0xFFFFFFFFLL, "target-cpu", 10);
  v5 = sub_1560340(a2 + 112, 0xFFFFFFFFLL, "target-cpu", 10);
  result = 0;
  if ( v4 == v5 )
  {
    v7 = sub_1560340(v3, 0xFFFFFFFFLL, "target-features", 15);
    return v7 == sub_1560340(a2 + 112, 0xFFFFFFFFLL, "target-features", 15);
  }
  return result;
}
