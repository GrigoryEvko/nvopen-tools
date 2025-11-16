// Function: sub_14E7710
// Address: 0x14e7710
//
__int64 __fastcall sub_14E7710(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  if ( a2 == 18 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x55545249565F5744LL | *(_QWORD *)(a1 + 8) ^ 0x6F6E5F5954494C41LL)
      && *(_WORD *)(a1 + 16) == 25966 )
    {
      return 0;
    }
    return 0xFFFFFFFFLL;
  }
  if ( a2 == 21 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x55545249565F5744LL | *(_QWORD *)(a1 + 8) ^ 0x69765F5954494C41LL)
      && *(_DWORD *)(a1 + 16) == 1635087474
      && *(_BYTE *)(a1 + 20) == 108 )
    {
      return 1;
    }
    return 0xFFFFFFFFLL;
  }
  result = 0xFFFFFFFFLL;
  if ( a2 == 26 )
  {
    if ( !(*(_QWORD *)a1 ^ 0x55545249565F5744LL | *(_QWORD *)(a1 + 8) ^ 0x75705F5954494C41LL)
      && *(_QWORD *)(a1 + 16) == 0x75747269765F6572LL
      && *(_WORD *)(a1 + 24) == 27745 )
    {
      return 2;
    }
    return 0xFFFFFFFFLL;
  }
  return result;
}
