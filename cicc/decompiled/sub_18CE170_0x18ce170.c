// Function: sub_18CE170
// Address: 0x18ce170
//
__int64 __fastcall sub_18CE170(__int64 a1, __int64 a2)
{
  bool v3; // al
  bool v4; // zf

  if ( !byte_4F99CA8[0] )
    return 0;
  v3 = sub_18C8CE0(a2);
  *(_BYTE *)(a1 + 344) = v3;
  if ( v3 )
  {
    v4 = *(_BYTE *)(a1 + 324) == 0;
    *(_QWORD *)(a1 + 312) = a2;
    if ( !v4 )
      *(_BYTE *)(a1 + 324) = 0;
    if ( *(_BYTE *)(a1 + 332) )
      *(_BYTE *)(a1 + 332) = 0;
    if ( *(_BYTE *)(a1 + 340) )
      *(_BYTE *)(a1 + 340) = 0;
    *(_QWORD *)(a1 + 232) = a2;
    *(_QWORD *)(a1 + 240) = 0;
    *(_QWORD *)(a1 + 248) = 0;
    *(_QWORD *)(a1 + 256) = 0;
    *(_QWORD *)(a1 + 264) = 0;
    *(_QWORD *)(a1 + 272) = 0;
    *(_QWORD *)(a1 + 280) = 0;
    *(_QWORD *)(a1 + 288) = 0;
    *(_QWORD *)(a1 + 296) = 0;
    *(_QWORD *)(a1 + 304) = 0;
  }
  return 0;
}
