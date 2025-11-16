// Function: sub_1A4FE10
// Address: 0x1a4fe10
//
bool __fastcall sub_1A4FE10(__int64 a1, __int64 *a2)
{
  __int64 v2; // rcx
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rdx

  v2 = a2[1];
  v3 = *a2;
  v4 = 24;
  if ( (_DWORD)v2 != -2 )
    v4 = 24LL * (unsigned int)(2 * v2 + 3);
  if ( (*(_BYTE *)(v3 + 23) & 0x40) != 0 )
  {
    v5 = *(_QWORD *)(*(_QWORD *)(v3 - 8) + v4);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      return *(_QWORD *)(*(_QWORD *)(a1 - 8) + 72LL) == v5;
  }
  else
  {
    v5 = *(_QWORD *)(v3 - 24LL * (*(_DWORD *)(v3 + 20) & 0xFFFFFFF) + v4);
    if ( (*(_BYTE *)(a1 + 23) & 0x40) != 0 )
      return *(_QWORD *)(*(_QWORD *)(a1 - 8) + 72LL) == v5;
  }
  return *(_QWORD *)(a1 - 24LL * (*(_DWORD *)(a1 + 20) & 0xFFFFFFF) + 72) == v5;
}
