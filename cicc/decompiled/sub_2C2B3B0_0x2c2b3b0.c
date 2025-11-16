// Function: sub_2C2B3B0
// Address: 0x2c2b3b0
//
__int64 __fastcall sub_2C2B3B0(_QWORD *a1, _QWORD *a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 result; // rax

  sub_C8CF70((__int64)a1, a1 + 4, 8, (__int64)(a2 + 4), (__int64)a2);
  v2 = a2[12];
  a2[12] = 0;
  a1[12] = v2;
  v3 = a2[13];
  a2[13] = 0;
  a1[13] = v3;
  result = a2[14];
  a2[14] = 0;
  a1[14] = result;
  return result;
}
