// Function: sub_8258E0
// Address: 0x8258e0
//
char *__fastcall sub_8258E0(__int64 a1, int a2)
{
  char *result; // rax
  int v3; // r12d

  result = "<unknown>";
  if ( a1 )
  {
    result = *(char **)(a1 + 8);
    if ( result )
    {
      v3 = unk_4F605C8;
      unk_4F605C8 = a2;
      if ( (*(_BYTE *)(a1 + 89) & 8) != 0 || (*(_BYTE *)(a1 + 198) & 0x20) != 0 && *result == 95 && result[1] == 90 )
        result = sub_8257B0(result);
      unk_4F605C8 = v3;
    }
    else
    {
      return "<unknown>";
    }
  }
  return result;
}
