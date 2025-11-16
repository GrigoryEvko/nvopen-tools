// Function: sub_C3BCA0
// Address: 0xc3bca0
//
__int64 __fastcall sub_C3BCA0(__int64 a1)
{
  unsigned int v1; // r13d
  _QWORD v3[8]; // [rsp+0h] [rbp-40h] BYREF

  v1 = 0;
  if ( (*(_BYTE *)(a1 + 20) & 6) == 0 )
    return 0;
  sub_C33EB0(v3, (__int64 *)a1);
  sub_C3BAB0((__int64)v3, 0);
  LOBYTE(v1) = (unsigned int)sub_C37950(a1, (__int64)v3) == 1;
  sub_C338F0((__int64)v3);
  return v1;
}
