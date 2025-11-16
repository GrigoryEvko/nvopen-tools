// Function: sub_10BBC70
// Address: 0x10bbc70
//
__int64 __fastcall sub_10BBC70(__int64 a1)
{
  unsigned int v1; // ebx
  unsigned __int64 v2; // rdx
  __int64 result; // rax

  v1 = *(_DWORD *)(a1 + 8);
  if ( v1 > 0x40 )
  {
    if ( (unsigned int)sub_C44630(a1) == 1 )
      return v1 - 1 - (unsigned int)sub_C444A0(a1);
    else
      return 0xFFFFFFFFLL;
  }
  else
  {
    v2 = *(_QWORD *)a1;
    result = 0xFFFFFFFFLL;
    if ( *(_QWORD *)a1 )
    {
      if ( (v2 & (v2 - 1)) == 0 )
      {
        _BitScanReverse64(&v2, v2);
        return -1 - (((unsigned int)v2 ^ 0x3F) - 64);
      }
    }
  }
  return result;
}
