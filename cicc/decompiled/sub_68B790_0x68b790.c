// Function: sub_68B790
// Address: 0x68b790
//
_BOOL8 __fastcall sub_68B790(unsigned __int64 a1)
{
  _BOOL8 result; // rax
  unsigned __int64 i; // rdx
  unsigned int v3; // edx
  __int64 v4; // rax

  result = 0;
  if ( (*(_BYTE *)(a1 + 175) & 7) != 0 )
  {
    for ( i = a1 >> 3; ; LODWORD(i) = v3 + 1 )
    {
      v3 = qword_4D03BF0[1] & i;
      v4 = *qword_4D03BF0 + 16LL * v3;
      if ( a1 == *(_QWORD *)v4 )
        return *(_DWORD *)(v4 + 8) != 0;
      if ( !*(_QWORD *)v4 )
        break;
    }
    return 0;
  }
  return result;
}
