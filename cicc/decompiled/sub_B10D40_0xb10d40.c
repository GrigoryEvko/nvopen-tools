// Function: sub_B10D40
// Address: 0xb10d40
//
__int64 __fastcall sub_B10D40(__int64 a1)
{
  __int64 v1; // rax
  unsigned __int8 v2; // dl
  __int64 v4; // rax

  v1 = sub_B10CD0(a1);
  v2 = *(_BYTE *)(v1 - 16);
  if ( (v2 & 2) != 0 )
  {
    if ( *(_DWORD *)(v1 - 24) != 2 )
      return 0;
    v4 = *(_QWORD *)(v1 - 32);
  }
  else
  {
    if ( ((*(_WORD *)(v1 - 16) >> 6) & 0xF) != 2 )
      return 0;
    v4 = v1 - 16 - 8LL * ((v2 >> 2) & 0xF);
  }
  return *(_QWORD *)(v4 + 8);
}
