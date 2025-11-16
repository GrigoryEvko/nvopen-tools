// Function: sub_2B098D0
// Address: 0x2b098d0
//
__int64 __fastcall sub_2B098D0(__int64 **a1, int *a2, unsigned __int64 a3, __int64 a4, unsigned __int64 a5, __int64 a6)
{
  __int64 v9; // rsi
  int v11; // edx
  _DWORD v12[9]; // [rsp+Ch] [rbp-24h] BYREF

  if ( a5 != 1 )
  {
    v9 = (unsigned int)(a5 < 2) + 6;
    return sub_2B097B0(*a1, v9, a6, a2, a3, 0, 0, 0);
  }
  v11 = *(_DWORD *)(*(_QWORD *)a4 + 432LL);
  if ( !v11 || !(unsigned __int8)sub_B4FCC0((__int64)a2, a3, v11, v12) )
  {
    v9 = 7;
    return sub_2B097B0(*a1, v9, a6, a2, a3, 0, 0, 0);
  }
  return 0;
}
