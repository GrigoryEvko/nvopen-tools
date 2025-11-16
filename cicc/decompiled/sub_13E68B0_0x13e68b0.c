// Function: sub_13E68B0
// Address: 0x13e68b0
//
__int64 __fastcall sub_13E68B0(__int64 a1, __int64 a2)
{
  __int64 v2; // r12

  v2 = *(_QWORD *)(a1 + 160);
  if ( *(_BYTE *)(v2 + 408) )
    return sub_13779F0(*(_QWORD *)(a1 + 160), a2);
  sub_137CAE0(*(_QWORD *)(a1 + 160), *(__int64 **)(v2 + 416), *(_QWORD *)(v2 + 424), *(_QWORD **)(v2 + 432));
  *(_BYTE *)(v2 + 408) = 1;
  return sub_13779F0(v2, a2);
}
