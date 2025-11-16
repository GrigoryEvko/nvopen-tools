// Function: sub_3249C40
// Address: 0x3249c40
//
__int64 __fastcall sub_3249C40(__int64 *a1, __int64 a2, __int16 a3, __int64 a4)
{
  int v7; // [rsp+Ch] [rbp-24h]

  BYTE2(v7) = 1;
  LOWORD(v7) = sub_3222A40(a1[26]);
  return sub_3249A20(a1, (unsigned __int64 **)(a2 + 8), a3, v7, a4);
}
