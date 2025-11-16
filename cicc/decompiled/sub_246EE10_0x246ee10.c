// Function: sub_246EE10
// Address: 0x246ee10
//
__int64 __fastcall sub_246EE10(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int8 v3; // dl
  __int64 v4; // r12

  v2 = *(_QWORD *)(a1 + 8);
  if ( !*(_DWORD *)(v2 + 4) )
    return 0;
  if ( !*(_BYTE *)(a1 + 633) )
    return sub_AD6530(*(_QWORD *)(v2 + 88), a2);
  v3 = *(_BYTE *)a2;
  v4 = a2;
  if ( *(_BYTE *)a2 <= 0x15u || v3 == 25 )
    return sub_AD6530(*(_QWORD *)(v2 + 88), a2);
  if ( v3 > 0x1Cu && (*(_BYTE *)(a2 + 7) & 0x20) != 0 )
  {
    a2 = 31;
    if ( sub_B91C10(v4, 31) )
    {
      v2 = *(_QWORD *)(a1 + 8);
      return sub_AD6530(*(_QWORD *)(v2 + 88), a2);
    }
  }
  return *sub_246EC10(a1 + 384, v4);
}
