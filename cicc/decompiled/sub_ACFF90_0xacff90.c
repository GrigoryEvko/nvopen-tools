// Function: sub_ACFF90
// Address: 0xacff90
//
bool __fastcall sub_ACFF90(__int64 a1, unsigned int a2)
{
  int v2; // r12d
  __int64 v3; // rbx
  __int64 *v4; // rdi

  v2 = 0;
  v3 = *(_QWORD *)(a1 + 16);
  if ( !v3 )
    return v2 == a2;
  while ( 1 )
  {
    while ( 1 )
    {
      v4 = *(__int64 **)(v3 + 24);
      if ( (unsigned __int8)(*(_BYTE *)v4 - 4) > 0x11u || !(unsigned __int8)sub_ACFEF0(v4, 0) )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( !v3 )
        return v2 == a2;
    }
    if ( a2 < ++v2 )
      break;
    v3 = *(_QWORD *)(v3 + 8);
    if ( !v3 )
      return v2 == a2;
  }
  return 0;
}
