// Function: sub_16C9F50
// Address: 0x16c9f50
//
bool __fastcall sub_16C9F50(__int64 a1, __int64 a2)
{
  _QWORD v3[2]; // [rsp+0h] [rbp-10h] BYREF

  v3[0] = a1;
  v3[1] = a2;
  return sub_16D23E0(v3, "()^$|*+?.[]\\{}", 14, 0) == -1;
}
