// Function: sub_1EEA050
// Address: 0x1eea050
//
unsigned __int64 __fastcall sub_1EEA050(__int64 a1, __int64 a2, __int64 a3, int a4, unsigned int a5, int a6)
{
  unsigned __int64 result; // rax
  __int64 v7; // rdx

  sub_1EE9CC0(a1, a2, a3, a4, a5, a6);
  sub_21049B0(a1 + 96, a2);
  result = a2 + 24;
  if ( *(_QWORD *)(a2 + 32) != a2 + 24 )
  {
    result = *(_QWORD *)(a2 + 24) & 0xFFFFFFFFFFFFFFF8LL;
    if ( !result )
      BUG();
    v7 = *(_QWORD *)result;
    if ( (*(_QWORD *)result & 4) == 0 && (*(_BYTE *)(result + 46) & 4) != 0 )
    {
      while ( 1 )
      {
        result = v7 & 0xFFFFFFFFFFFFFFF8LL;
        if ( (*(_BYTE *)((v7 & 0xFFFFFFFFFFFFFFF8LL) + 46) & 4) == 0 )
          break;
        v7 = *(_QWORD *)result;
      }
    }
    *(_QWORD *)(a1 + 32) = result;
    *(_BYTE *)(a1 + 44) = 1;
  }
  return result;
}
