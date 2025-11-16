// Function: sub_12255B0
// Address: 0x12255b0
//
__int64 __fastcall sub_12255B0(__int64 **a1, __int64 *a2, __int64 *a3)
{
  unsigned int v3; // r12d
  _BYTE *v5; // [rsp+8h] [rbp-28h] BYREF

  v3 = sub_12254B0((__int64)a1, &v5, a3);
  if ( !(_BYTE)v3 )
    *a2 = sub_B9F6F0(*a1, v5);
  return v3;
}
