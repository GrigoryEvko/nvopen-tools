// Function: sub_22A6BC0
// Address: 0x22a6bc0
//
__int64 __fastcall sub_22A6BC0(__int64 a1)
{
  unsigned int v1; // eax

  v1 = *(_DWORD *)(a1 + 12);
  if ( v1 <= 0xA )
  {
    if ( v1 )
      return 1;
LABEL_6:
    BUG();
  }
  if ( v1 - 11 > 7 )
    goto LABEL_6;
  return 0;
}
