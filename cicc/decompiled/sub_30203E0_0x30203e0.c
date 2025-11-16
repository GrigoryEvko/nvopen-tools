// Function: sub_30203E0
// Address: 0x30203e0
//
__int64 __fastcall sub_30203E0(_QWORD *a1)
{
  __int64 *v1; // rdi
  _QWORD v3[4]; // [rsp+0h] [rbp-40h] BYREF
  __int16 v4; // [rsp+20h] [rbp-20h]

  (*(void (__fastcall **)(_QWORD *))(*a1 + 192LL))(a1);
  sub_31EC4F0(a1);
  v1 = (__int64 *)a1[28];
  v4 = 261;
  v3[0] = "}\n";
  v3[1] = 2;
  sub_E99A90(v1, (__int64)v3);
  return 0;
}
