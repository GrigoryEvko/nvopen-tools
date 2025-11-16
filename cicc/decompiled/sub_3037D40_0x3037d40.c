// Function: sub_3037D40
// Address: 0x3037d40
//
bool __fastcall sub_3037D40(__int64 a1)
{
  int v1; // eax

  v1 = *(_DWORD *)(*(_QWORD *)(a1 + 537016) + 360LL);
  if ( v1 == -1 )
    return (*(_BYTE *)(*(_QWORD *)(a1 + 8) + 864LL) & 1) == 0;
  else
    return v1 == 1;
}
