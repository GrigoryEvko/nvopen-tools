// Function: sub_2434DB0
// Address: 0x2434db0
//
__int64 __fastcall sub_2434DB0(__int64 a1, __int64 *a2, unsigned __int64 a3, __int64 a4)
{
  unsigned int v4; // r12d
  __int64 v6[3]; // [rsp+8h] [rbp-18h] BYREF

  v6[0] = a3;
  v4 = sub_2433C90(a1, a3, a4);
  if ( (_BYTE)v4 )
    sub_2434B50(a2, v6);
  else
    sub_2434C80(a2, v6);
  return v4;
}
