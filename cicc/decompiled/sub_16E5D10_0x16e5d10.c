// Function: sub_16E5D10
// Address: 0x16e5d10
//
__int64 __fastcall sub_16E5D10(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  _QWORD v5[4]; // [rsp+0h] [rbp-20h] BYREF

  v3 = *a1;
  v5[1] = "0x%016llX";
  v5[2] = v3;
  v5[0] = &unk_49EFAE8;
  return sub_16E8450(a3, v5);
}
