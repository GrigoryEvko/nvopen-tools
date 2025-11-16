// Function: sub_6DF6A0
// Address: 0x6df6a0
//
__int64 __fastcall sub_6DF6A0(__int64 a1, _DWORD *a2)
{
  char v2; // al
  unsigned int v3; // r8d
  __int64 v5; // rdx

  v2 = *(_BYTE *)(a1 + 176);
  if ( v2 == 11 )
  {
    a1 = *(_QWORD *)(a1 + 184);
    v2 = *(_BYTE *)(a1 + 176);
  }
  if ( v2 != 2 )
  {
    if ( v2 == 3 || (v3 = 0, v2 == 13) )
    {
      *a2 = 1;
      return 1;
    }
    return v3;
  }
  v5 = *(_QWORD *)(a1 + 128);
  v3 = 1;
  if ( v5 == *(_QWORD *)&dword_4D03B80 )
    return v3;
  if ( *(_QWORD *)&dword_4D03B80 && v5 )
  {
    v3 = dword_4F07588;
    if ( dword_4F07588 )
      return (*(_QWORD *)(*(_QWORD *)&dword_4D03B80 + 32LL) == *(_QWORD *)(v5 + 32))
           & (unsigned __int8)(*(_QWORD *)(v5 + 32) != 0);
    return v3;
  }
  return 0;
}
