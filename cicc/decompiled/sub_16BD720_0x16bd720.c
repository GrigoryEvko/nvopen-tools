// Function: sub_16BD720
// Address: 0x16bd720
//
bool __fastcall sub_16BD720(__int64 a1, const void *a2, __int64 a3)
{
  _QWORD v4[2]; // [rsp+0h] [rbp-10h] BYREF

  v4[0] = *(_QWORD *)a1;
  v4[1] = *(unsigned int *)(a1 + 8);
  return sub_16BD3B0((__int64)v4, a2, a3);
}
