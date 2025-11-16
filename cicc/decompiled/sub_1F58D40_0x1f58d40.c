// Function: sub_1F58D40
// Address: 0x1f58d40
//
__int64 __fastcall sub_1F58D40(__int64 a1)
{
  __int64 v1; // rbx

  v1 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v1 + 8) == 11 )
    return *(_DWORD *)(v1 + 8) >> 8;
  else
    return *(_DWORD *)(v1 + 32) * (unsigned int)sub_1643030(*(_QWORD *)(v1 + 24));
}
