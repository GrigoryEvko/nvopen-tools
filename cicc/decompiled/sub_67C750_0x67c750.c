// Function: sub_67C750
// Address: 0x67c750
//
__int64 sub_67C750()
{
  __int64 result; // rax
  char *v1; // rax

  if ( getenv("NOCOLOR") || !(unsigned int)sub_7216D0() )
  {
    dword_4F073C8 = 0;
    result = 0;
LABEL_3:
    unk_4F073CC = result;
    return result;
  }
  result = dword_4F073C8;
  if ( !dword_4F073C8 )
    goto LABEL_3;
  unk_4F073C0 = getenv("EDG_COLORS");
  if ( !unk_4F073C0 )
  {
    v1 = getenv("GCC_COLORS");
    if ( !v1 )
      v1 = "error=01;31:warning=01;35:note=01;36:locus=01:quote=01:range1=32";
    unk_4F073C0 = v1;
  }
  sub_67B560(2u, "error");
  sub_67B560(3u, "warning");
  sub_67B560(4u, "note");
  sub_67B560(5u, "locus");
  sub_67B560(6u, "quote");
  sub_67B560(7u, "range1");
  unk_4F073CC = dword_4F073C8;
  return dword_4F073C8;
}
