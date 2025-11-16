// Function: sub_2739680
// Address: 0x2739680
//
unsigned __int64 __fastcall sub_2739680(__int64 a1)
{
  unsigned __int64 result; // rax
  __int64 v2; // rdx
  unsigned __int64 v3; // rax
  int v4; // edx

  result = *(_QWORD *)(a1 + 24);
  if ( *(_BYTE *)result == 84 )
  {
    v2 = *(_QWORD *)(*(_QWORD *)(result - 8)
                   + 32LL * *(unsigned int *)(result + 72)
                   + 8LL * (unsigned int)((a1 - *(_QWORD *)(result - 8)) >> 5));
    v3 = *(_QWORD *)(v2 + 48) & 0xFFFFFFFFFFFFFFF8LL;
    if ( v3 == v2 + 48 )
    {
      return 0;
    }
    else
    {
      if ( !v3 )
        BUG();
      v4 = *(unsigned __int8 *)(v3 - 24);
      result = v3 - 24;
      if ( (unsigned int)(v4 - 30) >= 0xB )
        return 0;
    }
  }
  return result;
}
