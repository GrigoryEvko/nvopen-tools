// Function: sub_122E8D0
// Address: 0x122e8d0
//
__int64 __fastcall sub_122E8D0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r12d
  unsigned int v4; // [rsp+4h] [rbp-1Ch] BYREF
  __int64 v5[3]; // [rsp+8h] [rbp-18h] BYREF

  v2 = sub_122E830(a1, &v4, v5);
  if ( !(_BYTE)v2 )
    sub_B994D0(a2, v4, v5[0]);
  return v2;
}
