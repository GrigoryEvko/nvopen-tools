// Function: sub_8DD040
// Address: 0x8dd040
//
__int64 __fastcall sub_8DD040(__int64 a1, _DWORD *a2)
{
  __int64 v2; // r12
  __int64 v4; // r14

  if ( !unk_4F072F4 )
    return 0;
  v2 = a1;
  if ( !a1 )
    return 0;
  while ( *(_BYTE *)(v2 + 140) != 13 )
  {
    v2 = sub_8D48B0(v2, 0);
    if ( !v2 )
      return 0;
  }
  v4 = *(_QWORD *)(v2 + 168);
  if ( !(unsigned int)sub_8DD010(v4) )
    return 0;
  sub_685360(0x57Cu, a2, v4);
  sub_725570(v2, 0);
  return 1;
}
