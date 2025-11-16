// Function: sub_2AA0010
// Address: 0x2aa0010
//
bool __fastcall sub_2AA0010(__int64 a1, __int64 a2)
{
  _BYTE *v3; // r13
  unsigned __int64 v4; // rax
  char v5; // dl
  __int64 v6; // rdx
  __int64 v7; // rax

  if ( *(_BYTE *)a2 != 31 )
    return 0;
  if ( (*(_DWORD *)(a2 + 4) & 0x7FFFFFF) != 3 )
    return 0;
  v3 = *(_BYTE **)(a2 - 96);
  if ( *v3 != 82 )
    return 0;
  v4 = sub_B53900(*(_QWORD *)(a2 - 96));
  sub_B53630(v4, *(_QWORD *)a1);
  if ( !v5 )
    return 0;
  if ( *((_QWORD *)v3 - 8) != *(_QWORD *)(a1 + 8) )
    return 0;
  v6 = *((_QWORD *)v3 - 4);
  if ( !v6 )
    return 0;
  **(_QWORD **)(a1 + 16) = v6;
  v7 = *(_QWORD *)(a2 - 32);
  if ( !v7 )
    return 0;
  **(_QWORD **)(a1 + 24) = v7;
  return *(_QWORD *)(a2 - 64) == *(_QWORD *)(a1 + 32);
}
