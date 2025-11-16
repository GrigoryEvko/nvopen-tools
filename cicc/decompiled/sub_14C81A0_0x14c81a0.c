// Function: sub_14C81A0
// Address: 0x14c81a0
//
__int64 __fastcall sub_14C81A0(__int64 a1)
{
  unsigned __int8 v1; // dl
  __int64 result; // rax
  unsigned int v3; // edx

  v1 = *(_BYTE *)(a1 + 16);
  result = 4;
  if ( v1 > 3u )
  {
    if ( v1 == 17 )
    {
      if ( (unsigned __int8)sub_15E04B0(a1) || *(_BYTE *)(*(_QWORD *)a1 + 8LL) != 15 )
      {
        return 0;
      }
      else
      {
        v3 = *(_DWORD *)(a1 + 32);
        result = 2;
        if ( v3 <= 0x1B )
          return 1LL << ((unsigned __int8)v3 + 4);
      }
    }
    else
    {
      return 0;
    }
  }
  return result;
}
