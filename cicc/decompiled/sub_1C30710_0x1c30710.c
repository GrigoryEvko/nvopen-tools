// Function: sub_1C30710
// Address: 0x1c30710
//
__int64 __fastcall sub_1C30710(__int64 a1)
{
  __int64 v2; // rax

  if ( *(_BYTE *)(a1 + 16) == 78
    && (v2 = *(_QWORD *)(a1 - 24), !*(_BYTE *)(v2 + 16))
    && (*(_BYTE *)(v2 + 33) & 0x20) != 0 )
  {
    return sub_1C30530(*(_DWORD *)(v2 + 36));
  }
  else
  {
    return 0;
  }
}
