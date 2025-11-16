// Function: sub_66A6A0
// Address: 0x66a6a0
//
void __fastcall sub_66A6A0(__int64 a1)
{
  char v1; // al

  if ( *(_QWORD *)(a1 + 8) && (v1 = *(_BYTE *)(a1 + 89), (v1 & 1) == 0) )
  {
    if ( (v1 & 4) == 0 || dword_4F077BC )
    {
      if ( !qword_4D0495C || dword_4F04C34 )
        *(_BYTE *)(a1 + 88) = *(_BYTE *)(a1 + 88) & 0x8F | 0x20;
      else
        *(_BYTE *)(a1 + 88) = *(_BYTE *)(a1 + 88) & 0x8F | 0x10;
    }
    else
    {
      *(_BYTE *)(a1 + 88) = *(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 88LL) & 0x70
                          | *(_BYTE *)(a1 + 88) & 0x8F;
    }
  }
  else
  {
    *(_BYTE *)(a1 + 88) &= 0x8Fu;
  }
}
