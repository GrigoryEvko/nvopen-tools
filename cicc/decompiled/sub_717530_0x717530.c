// Function: sub_717530
// Address: 0x717530
//
__int64 __fastcall sub_717530(__int64 a1, _QWORD *a2)
{
  __int64 result; // rax
  __int64 v3; // r13
  __int64 v4; // rax

  if ( a2 )
    *a2 = 0;
  if ( *(_BYTE *)(a1 + 173) != 6 )
    return 0;
  if ( *(_BYTE *)(a1 + 176) != 2 )
    return 0;
  if ( *(_QWORD *)(a1 + 192) )
    return 0;
  if ( (*(_BYTE *)(a1 + 168) & 8) == 0 )
    return 0;
  v3 = *(_QWORD *)(a1 + 184);
  if ( *(_BYTE *)(v3 + 173) != 2 )
    return 0;
  v4 = sub_8D67C0(*(_QWORD *)(v3 + 128));
  if ( !sub_70D540(*(_QWORD *)(a1 + 128), v4) )
    return 0;
  result = 1;
  if ( a2 )
    *a2 = v3;
  return result;
}
