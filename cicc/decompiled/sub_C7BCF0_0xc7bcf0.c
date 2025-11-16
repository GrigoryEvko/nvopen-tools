// Function: sub_C7BCF0
// Address: 0xc7bcf0
//
__int64 __fastcall sub_C7BCF0(__int64 a1, __int64 *a2)
{
  if ( *(_DWORD *)(a1 + 8) > 0x40u )
  {
    sub_C43BD0((_QWORD *)a1, a2);
    if ( *(_DWORD *)(a1 + 24) <= 0x40u )
      goto LABEL_3;
  }
  else
  {
    *(_QWORD *)a1 |= *a2;
    if ( *(_DWORD *)(a1 + 24) <= 0x40u )
    {
LABEL_3:
      *(_QWORD *)(a1 + 16) &= a2[2];
      return a1;
    }
  }
  sub_C43B90((_QWORD *)(a1 + 16), a2 + 2);
  return a1;
}
