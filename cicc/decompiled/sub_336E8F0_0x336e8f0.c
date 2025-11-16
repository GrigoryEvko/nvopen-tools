// Function: sub_336E8F0
// Address: 0x336e8f0
//
__int64 __fastcall sub_336E8F0(__int64 a1, __int64 a2, int a3)
{
  __int64 result; // rax
  __int64 v4; // rsi

  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 8) = a3;
  if ( a2 )
  {
    result = a2 + 48;
    if ( a1 != a2 + 48 )
    {
      v4 = *(_QWORD *)(a2 + 48);
      *(_QWORD *)a1 = v4;
      if ( v4 )
        return sub_B96E90(a1, v4, 1);
    }
  }
  return result;
}
