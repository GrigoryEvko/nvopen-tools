// Function: sub_2D58330
// Address: 0x2d58330
//
bool __fastcall sub_2D58330(__int64 *a1, __int64 a2)
{
  __int64 v2; // rbp
  __int64 v3; // r8
  unsigned int v5; // [rsp-Ch] [rbp-Ch] BYREF
  __int64 v6; // [rsp-8h] [rbp-8h]

  v3 = *(_QWORD *)(a2 - 32);
  if ( !v3 || *(_BYTE *)v3 || *(_QWORD *)(v3 + 24) != *(_QWORD *)(a2 + 80) )
    return 0;
  if ( (*(_BYTE *)(v3 + 33) & 0x20) != 0 )
  {
    if ( (unsigned int)(*(_DWORD *)(v3 + 36) - 238) <= 5 )
      return ((1LL << (*(_BYTE *)(v3 + 36) + 18)) & 0x29) != 0;
    return 0;
  }
  if ( !a1 )
    return 0;
  v6 = v2;
  return sub_981210(*a1, v3, &v5) && v5 - 459 <= 0xD && ((1LL << ((unsigned __int8)v5 + 53)) & 0x2811) != 0;
}
