// Function: sub_300C0A0
// Address: 0x300c0a0
//
bool __fastcall sub_300C0A0(_QWORD *a1, int a2)
{
  __int64 v2; // rsi
  bool result; // al
  __int64 v4; // rdx
  int v5; // edx

  v2 = a2 & 0x7FFFFFFF;
  result = 0;
  if ( (unsigned int)v2 < *(_DWORD *)(*a1 + 248LL) )
  {
    v4 = *(_QWORD *)(*a1 + 240LL) + 40 * v2;
    if ( *(_DWORD *)(v4 + 16) )
    {
      v5 = **(_DWORD **)(v4 + 8);
      result = 1;
      if ( (unsigned int)(v5 - 1) > 0x3FFFFFFE )
        return v5 < 0 && *(_DWORD *)(a1[4] + 4LL * (v5 & 0x7FFFFFFF)) != 0;
    }
  }
  return result;
}
