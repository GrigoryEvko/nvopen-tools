// Function: sub_3982020
// Address: 0x3982020
//
__int64 __fastcall sub_3982020(unsigned __int64 *a1, __int64 a2, unsigned __int16 a3)
{
  unsigned int v4; // r13d
  unsigned __int64 v5; // rdi
  _BYTE v7[50]; // [rsp+Eh] [rbp-32h] BYREF

  v4 = 0;
  if ( a2 )
  {
    LOWORD(v4) = sub_3971A70(a2);
    v4 = ((unsigned __int8)sub_396E560(a2) << 16) | v4 & 0xFF00FFFF;
  }
  sub_14E99E0(v7, a3, v4);
  if ( v7[1] )
    return v7[0];
  v5 = *a1;
  if ( a3 > 0xEu )
    return sub_3946290(v5);
  else
    return sub_39462B0(v5);
}
