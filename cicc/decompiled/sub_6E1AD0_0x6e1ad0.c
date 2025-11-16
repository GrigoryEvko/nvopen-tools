// Function: sub_6E1AD0
// Address: 0x6e1ad0
//
__int64 __fastcall sub_6E1AD0(__int64 a1, _QWORD *a2)
{
  __int64 v3; // r12

  if ( *(_BYTE *)(a1 + 8) )
    return 0;
  v3 = *(_QWORD *)(a1 + 24);
  if ( !sub_694910((_BYTE *)(v3 + 8)) )
    return 0;
  *a2 = v3 + 152;
  return 1;
}
