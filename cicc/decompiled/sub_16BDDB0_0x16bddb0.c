// Function: sub_16BDDB0
// Address: 0x16bddb0
//
unsigned __int64 __fastcall sub_16BDDB0(__int64 a1)
{
  _QWORD v2[2]; // [rsp+0h] [rbp-10h] BYREF

  v2[0] = *(_QWORD *)a1;
  v2[1] = *(unsigned int *)(a1 + 8);
  return sub_16BDD90((__int64)v2);
}
