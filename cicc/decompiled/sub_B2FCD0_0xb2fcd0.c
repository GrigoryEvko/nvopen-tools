// Function: sub_B2FCD0
// Address: 0xb2fcd0
//
__int64 __fastcall sub_B2FCD0(__int64 a1)
{
  __int64 result; // rax
  char v2; // al
  __int64 v3; // rax
  int v4; // edx

  if ( (*(_BYTE *)(a1 + 32) & 0xF) == 1 )
    return 0;
  if ( sub_B2FC80(a1) )
    return 0;
  v2 = *(_BYTE *)(a1 + 32) & 0xF;
  if ( ((v2 + 14) & 0xFu) <= 3
    || ((v2 + 7) & 0xFu) <= 1
    || (*(_WORD *)(a1 + 34) & 0x400) != 0 && ((*(_WORD *)(a1 + 34) >> 1) & 0x3F) != 0 )
  {
    return 0;
  }
  v3 = *(_QWORD *)(a1 + 40);
  if ( !v3 )
  {
    if ( (*(_BYTE *)(a1 + 33) & 0x40) != 0 )
    {
LABEL_12:
      result = 1;
      if ( *(_BYTE *)a1 == 3 )
        return (unsigned int)sub_A73380((__int64 *)(a1 + 72), "toc-data", 8u) ^ 1;
      return result;
    }
    return 0;
  }
  v4 = *(_DWORD *)(v3 + 284);
  if ( v4 == 3 )
    return (*(_BYTE *)(a1 + 33) & 0x40) != 0;
  result = 1;
  if ( v4 == 8 )
    goto LABEL_12;
  return result;
}
