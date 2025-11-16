// Function: sub_76FE50
// Address: 0x76fe50
//
__int64 sub_76FE50()
{
  __int64 result; // rax
  unsigned int v1; // esi
  int v2; // edi
  int v3; // r9d
  unsigned int v4; // r8d
  char v5; // cl

  result = *(_QWORD *)(unk_4F07288 + 144LL);
  if ( result )
  {
    v1 = dword_4F068EC;
    v2 = dword_4F077C4;
    v3 = unk_4F07778;
    v4 = dword_4F077C0;
    while ( 1 )
    {
      if ( *(char *)(result + 192) >= 0 )
        goto LABEL_3;
      v5 = *(_BYTE *)(result + 203);
      if ( (v5 & 2) == 0 )
        goto LABEL_3;
      if ( (*(_BYTE *)(result + 192) & 1) != 0 )
        *(_BYTE *)(result + 204) |= 1u;
      if ( !*(_BYTE *)(result + 172) )
      {
        if ( !v1 && (v2 == 2 || v3 <= 199900) )
        {
          if ( !v4 || (v5 & 0x40) != 0 )
            goto LABEL_13;
LABEL_16:
          *(_BYTE *)(result + 204) |= 1u;
          goto LABEL_13;
        }
        if ( (v5 & 0x40) == 0 )
          goto LABEL_16;
      }
LABEL_13:
      if ( (*(_BYTE *)(result + 204) & 1) != 0 )
      {
LABEL_3:
        result = *(_QWORD *)(result + 112);
        if ( !result )
          return result;
      }
      else
      {
        *(_BYTE *)(result + 88) &= ~4u;
        result = *(_QWORD *)(result + 112);
        if ( !result )
          return result;
      }
    }
  }
  return result;
}
