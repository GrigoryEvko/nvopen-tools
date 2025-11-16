// Function: sub_CC7280
// Address: 0xcc7280
//
__int64 __fastcall sub_CC7280(__int64 *a1)
{
  __int64 v1; // rax
  char v3; // [rsp+Fh] [rbp-11h] BYREF
  __int64 v4[2]; // [rsp+10h] [rbp-10h] BYREF

  v1 = *a1;
  v3 = 45;
  v4[0] = v1;
  v4[1] = a1[1];
  sub_C931B0(v4, &v3, 1u, 0);
  return v4[0];
}
