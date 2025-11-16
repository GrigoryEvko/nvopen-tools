// Function: sub_300C040
// Address: 0x300c040
//
bool __fastcall sub_300C040(_QWORD *a1, int a2)
{
  __int64 v2; // rsi
  bool result; // al
  __int64 v4; // rcx
  int v5; // edx
  __int64 v6; // rax

  v2 = a2 & 0x7FFFFFFF;
  result = 0;
  if ( (unsigned int)v2 < *(_DWORD *)(*a1 + 248LL) )
  {
    v4 = *(_QWORD *)(*a1 + 240LL) + 40 * v2;
    if ( *(_DWORD *)(v4 + 16) )
    {
      v5 = **(_DWORD **)(v4 + 8);
      if ( !*(_DWORD *)v4 )
      {
        if ( v5 )
        {
          v6 = a1[4];
          if ( v5 < 0 )
            v5 = *(_DWORD *)(v6 + 4LL * (v5 & 0x7FFFFFFF));
          return *(_DWORD *)(v6 + 4 * v2) == v5;
        }
      }
    }
  }
  return result;
}
