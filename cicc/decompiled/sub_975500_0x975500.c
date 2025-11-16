// Function: sub_975500
// Address: 0x975500
//
__int64 __fastcall sub_975500(unsigned int a1, __int64 a2, __int64 a3, _QWORD *a4, _BYTE *a5)
{
  __int64 v6[2]; // [rsp+0h] [rbp-10h] BYREF

  if ( a5 )
  {
    if ( (unsigned __int8)(*a5 - 34) > 0x33u )
    {
      a5 = 0;
    }
    else if ( ((0x8000000000041uLL >> (*a5 - 34)) & 1) == 0 )
    {
      a5 = 0;
    }
  }
  v6[1] = a3;
  v6[0] = a2;
  return sub_9732F0(a1, a4, v6, (__int64)a5);
}
