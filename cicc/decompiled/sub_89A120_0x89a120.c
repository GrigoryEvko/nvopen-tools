// Function: sub_89A120
// Address: 0x89a120
//
__int64 __fastcall sub_89A120(__int64 a1)
{
  __int64 result; // rax
  char v2; // dl

  result = unk_4D03FD8;
  if ( unk_4D03FD8 )
  {
    v2 = *(_BYTE *)(a1 + 195);
    result = 0;
    if ( (v2 & 8) == 0 && (v2 & 3) != 1 && (*(char *)(a1 + 192) >= 0 || *(_BYTE *)(a1 + 172) == 2 && !unk_4D03B70) )
    {
      result = 1;
      if ( (*(_BYTE *)(a1 + 193) & 0x10) != 0 && (*(_BYTE *)(a1 + 89) & 4) != 0 )
        return (*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 176LL) & 0x11000) != 4096;
    }
  }
  return result;
}
