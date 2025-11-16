// Function: sub_2AB26E0
// Address: 0x2ab26e0
//
__int64 __fastcall sub_2AB26E0(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v5; // [rsp+10h] [rbp-8h]

  BYTE4(v5) = BYTE4(a3);
  LODWORD(v5) = a4 * a3;
  return sub_B33F10(a1, a2, v5);
}
