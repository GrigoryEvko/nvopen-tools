// Function: sub_1B43680
// Address: 0x1b43680
//
bool __fastcall sub_1B43680(__int64 a1)
{
  __int64 v1; // rax
  _BYTE *v2; // rdi
  __int64 v3; // rcx
  bool result; // al
  __int64 v5; // rdx
  int v6; // eax

  if ( !*(_QWORD *)(a1 + 48) && *(__int16 *)(a1 + 18) >= 0 )
    return 0;
  v1 = sub_1625790(a1, 2);
  if ( !v1 )
    return 0;
  v2 = *(_BYTE **)(v1 - 8LL * *(unsigned int *)(v1 + 8));
  if ( !v2 || *v2 )
    return 0;
  v3 = sub_161E970((__int64)v2);
  result = 0;
  if ( v5 == 14 )
  {
    if ( *(_QWORD *)v3 != 0x775F68636E617262LL
      || *(_DWORD *)(v3 + 8) != 1751607653
      || (v6 = 0, *(_WORD *)(v3 + 12) != 29556) )
    {
      v6 = 1;
    }
    return v6 == 0;
  }
  return result;
}
