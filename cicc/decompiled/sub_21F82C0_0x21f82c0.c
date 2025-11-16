// Function: sub_21F82C0
// Address: 0x21f82c0
//
__int64 __fastcall sub_21F82C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  __int64 result; // rax
  __int64 v6; // r11
  __int64 v7; // rsi

  result = a2 - 1;
  v6 = (a2 - 1) / 2;
  if ( a2 > a3 )
  {
    while ( 1 )
    {
      v7 = a1 + 16 * a2;
      result = a1 + 16 * v6;
      if ( *(_DWORD *)(result + 8) >= a5 )
        break;
      *(_QWORD *)v7 = *(_QWORD *)result;
      *(_DWORD *)(v7 + 8) = *(_DWORD *)(result + 8);
      a2 = v6;
      if ( a3 >= v6 )
      {
        *(_QWORD *)result = a4;
        *(_DWORD *)(result + 8) = a5;
        return result;
      }
      v6 = (v6 - 1) / 2;
    }
  }
  else
  {
    v7 = a1 + 16 * a2;
  }
  *(_QWORD *)v7 = a4;
  *(_DWORD *)(v7 + 8) = a5;
  return result;
}
