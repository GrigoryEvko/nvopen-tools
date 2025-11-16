// Function: sub_6DF870
// Address: 0x6df870
//
__int64 __fastcall sub_6DF870(__int64 a1)
{
  unsigned int v1; // r12d

  if ( (unsigned int)sub_8D2660(*(_QWORD *)a1) )
    return 1;
  v1 = 0;
  if ( *(_BYTE *)(a1 + 16) != 2 )
    return v1;
  v1 = sub_712570(a1 + 144);
  if ( v1 )
  {
    return 1;
  }
  else if ( dword_4F077BC
         && *(_BYTE *)(a1 + 16) == 2
         && *(_BYTE *)(a1 + 317) == 1
         && (*(_BYTE *)(a1 + 313) & 4) == 0
         && (unsigned int)sub_8D29A0(*(_QWORD *)(a1 + 272)) )
  {
    return (unsigned int)sub_6210B0(a1 + 144, 0) == 0;
  }
  return v1;
}
