// Function: sub_5D4D90
// Address: 0x5d4d90
//
__int64 __fastcall sub_5D4D90(__int64 a1, _DWORD *a2)
{
  __int64 i; // rax

  if ( !(unsigned int)sub_8D2E30(a1) )
    return 0;
  for ( i = sub_8D46C0(a1); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  if ( (*(_BYTE *)(i + 142) & 0x10) == 0 )
    return 0;
  if ( !*a2 )
  {
    *a2 = 1;
    putc(40, stream);
    ++dword_4CF7F40;
  }
  sub_5D3E00(a1);
  putc(40, stream);
  ++dword_4CF7F40;
  return 1;
}
