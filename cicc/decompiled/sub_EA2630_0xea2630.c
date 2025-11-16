// Function: sub_EA2630
// Address: 0xea2630
//
__int64 __fastcall sub_EA2630(__int64 a1, int a2)
{
  int v3; // [rsp+Ch] [rbp-14h] BYREF
  _QWORD v4[2]; // [rsp+10h] [rbp-10h] BYREF

  v3 = a2;
  v4[0] = a1;
  v4[1] = &v3;
  return sub_ECE300(a1, sub_EAD990, v4, 1);
}
