// Function: sub_1F5BE90
// Address: 0x1f5be90
//
bool __fastcall sub_1F5BE90(__int64 a1, int a2)
{
  __int64 v2; // rdx
  bool result; // al
  int v4; // edx

  v2 = *(_QWORD *)(*(_QWORD *)(a1 + 232) + 208LL) + 40LL * (a2 & 0x7FFFFFFF);
  result = 0;
  if ( *(_DWORD *)(v2 + 16) )
  {
    v4 = **(_DWORD **)(v2 + 8);
    result = 1;
    if ( v4 <= 0 )
      return v4 && *(_DWORD *)(*(_QWORD *)(a1 + 264) + 4LL * (v4 & 0x7FFFFFFF)) != 0;
  }
  return result;
}
