// Function: sub_38D4AB0
// Address: 0x38d4ab0
//
void __fastcall sub_38D4AB0(__int64 a1, __int64 a2)
{
  if ( -858993459 * (unsigned int)((__int64)(*(_QWORD *)(a1 + 32) - *(_QWORD *)(a1 + 24)) >> 4) )
  {
    if ( *(_BYTE *)(a1 + 280) )
    {
      sub_38C7400((__int64 *)a1, a2, 1u);
      if ( !*(_BYTE *)(a1 + 281) )
        return;
    }
    else if ( !*(_BYTE *)(a1 + 281) )
    {
      return;
    }
    sub_38C7400((__int64 *)a1, a2, 0);
  }
}
