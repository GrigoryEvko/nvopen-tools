// Function: sub_C65690
// Address: 0xc65690
//
bool __fastcall sub_C65690(__int64 a1, const void *a2, __int64 a3)
{
  _QWORD v4[2]; // [rsp+0h] [rbp-10h] BYREF

  v4[0] = *(_QWORD *)a1;
  v4[1] = *(unsigned int *)(a1 + 8);
  return sub_C65390((__int64)v4, a2, a3);
}
