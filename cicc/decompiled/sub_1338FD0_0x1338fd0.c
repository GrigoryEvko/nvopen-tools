// Function: sub_1338FD0
// Address: 0x1338fd0
//
__int64 __fastcall sub_1338FD0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6, __int64 a7)
{
  __int64 result; // rax
  unsigned __int64 v8; // rsi

  result = 1;
  if ( !(a7 | a6 | a5 | a4) )
  {
    v8 = *(_QWORD *)(a2 + 8);
    result = 14;
    if ( v8 <= 0xFFFFFFFF )
    {
      sub_1338E30(a1, v8, 1u);
      return 0;
    }
  }
  return result;
}
