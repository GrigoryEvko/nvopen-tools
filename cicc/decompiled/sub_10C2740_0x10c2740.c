// Function: sub_10C2740
// Address: 0x10c2740
//
__int64 __fastcall sub_10C2740(_QWORD **a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r9
  __int64 v4; // rsi
  __int64 v5; // r8
  int v6; // r10d
  int v7; // r11d

  result = 0;
  if ( *(_BYTE *)a2 == 78 )
  {
    v3 = *(_QWORD *)(a2 - 32);
    v4 = *(_QWORD *)(a2 + 8);
    v5 = *(_QWORD *)(v3 + 8);
    v6 = *(unsigned __int8 *)(v4 + 8);
    v7 = *(unsigned __int8 *)(v5 + 8);
    if ( (unsigned int)(v6 - 17) <= 1 == (unsigned int)(v7 - 17) <= 1 )
    {
      if ( (unsigned int)(v7 - 17) > 1
        || *(_DWORD *)(v5 + 32) == *(_DWORD *)(v4 + 32) && ((_BYTE)v7 == 18) == ((_BYTE)v6 == 18) )
      {
        **a1 = v3;
        return 1;
      }
      else
      {
        return 0;
      }
    }
  }
  return result;
}
