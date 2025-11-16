// Function: sub_38A2200
// Address: 0x38a2200
//
__int64 __fastcall sub_38A2200(__int64 **a1, __int64 *a2, __int64 *a3, double a4, double a5, double a6)
{
  unsigned int v6; // r12d
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v6 = sub_38A2140((__int64)a1, v8, a3, a4, a5, a6);
  if ( !(_BYTE)v6 )
    *a2 = sub_1628DA0(*a1, v8[0]);
  return v6;
}
