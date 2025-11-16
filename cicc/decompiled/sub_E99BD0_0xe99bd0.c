// Function: sub_E99BD0
// Address: 0xe99bd0
//
__int64 __fastcall sub_E99BD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 *v3; // rdi
  _QWORD v5[4]; // [rsp+0h] [rbp-30h] BYREF
  __int16 v6; // [rsp+20h] [rbp-10h]

  v3 = *(__int64 **)(a1 + 8);
  v5[0] = a2;
  v6 = 261;
  v5[1] = a3;
  return sub_E99A90(v3, (__int64)v5);
}
