// Function: sub_5C9570
// Address: 0x5c9570
//
__int64 __fastcall sub_5C9570(__int64 a1, __int64 a2)
{
  __int64 v2; // rax

  if ( !(unsigned int)sub_88B6F0() )
  {
    v2 = **(_QWORD **)(a1 + 48);
    if ( v2 )
      *(_BYTE *)(v2 + 83) |= 0x40u;
  }
  return a2;
}
