// Function: sub_5D48C0
// Address: 0x5d48c0
//
int __fastcall sub_5D48C0(__int64 a1)
{
  int result; // eax
  __int64 v2; // rax

  result = sub_8D2310(a1);
  if ( !result )
  {
    if ( (unsigned int)sub_8D3410(a1) )
    {
      v2 = sub_8D40F0(a1);
      if ( *(_BYTE *)(v2 + 140) == 12 && (sub_8D4C10(v2, 1) & 1) != 0 )
        sub_5D4810(a1);
    }
    result = putc(38, stream);
    ++dword_4CF7F40;
  }
  return result;
}
