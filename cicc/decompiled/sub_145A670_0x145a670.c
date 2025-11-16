// Function: sub_145A670
// Address: 0x145a670
//
__int64 __fastcall sub_145A670(__int64 a1, __int64 a2)
{
  _BYTE v3[2]; // [rsp+Eh] [rbp-2h] BYREF

  v3[0] = 0;
  sub_145A080(a2, v3);
  return v3[0] ^ 1u;
}
