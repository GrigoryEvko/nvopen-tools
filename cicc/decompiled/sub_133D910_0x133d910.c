// Function: sub_133D910
// Address: 0x133d910
//
__int64 __fastcall sub_133D910(__int64 a1, __int64 *a2, __int64 a3)
{
  unsigned int v4; // r12d

  v4 = sub_130AF40(a1);
  if ( !(_BYTE)v4 )
  {
    *(_BYTE *)(a1 + 112) = 0;
    sub_133D860((__int64 *)a1, a2, a3);
  }
  return v4;
}
