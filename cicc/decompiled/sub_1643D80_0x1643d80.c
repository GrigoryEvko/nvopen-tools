// Function: sub_1643D80
// Address: 0x1643d80
//
__int64 __fastcall sub_1643D80(__int64 a1, unsigned int a2)
{
  if ( *(_BYTE *)(a1 + 8) == 13 )
    return *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL * a2);
  else
    return *(_QWORD *)(a1 + 24);
}
