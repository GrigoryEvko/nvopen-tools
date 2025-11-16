// Function: sub_39B19F0
// Address: 0x39b19f0
//
bool __fastcall sub_39B19F0(__int64 a1, __int64 a2, __int64 a3)
{
  _QWORD *v3; // r13
  __int64 v4; // rbx
  __int64 v5; // r8
  bool result; // al
  __int64 v7; // rbx

  v3 = (_QWORD *)(a3 + 112);
  v4 = sub_1560340((_QWORD *)(a3 + 112), -1, "target-cpu", 0xAu);
  v5 = sub_1560340((_QWORD *)(a2 + 112), -1, "target-cpu", 0xAu);
  result = 0;
  if ( v4 == v5 )
  {
    v7 = sub_1560340(v3, -1, "target-features", 0xFu);
    return v7 == sub_1560340((_QWORD *)(a2 + 112), -1, "target-features", 0xFu);
  }
  return result;
}
