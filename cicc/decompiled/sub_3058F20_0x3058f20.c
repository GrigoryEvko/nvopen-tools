// Function: sub_3058F20
// Address: 0x3058f20
//
char *__fastcall sub_3058F20(__int16 ***a1)
{
  char *result; // rax

  if ( a1 == &off_4A2FB60 )
    return ".b32";
  if ( a1 == &off_4A2F980 )
    return ".b64";
  if ( a1 == &off_4A2F8C0 )
    return ".b128";
  if ( a1 == &off_4A2FA40 )
    return ".b64";
  if ( a1 == &off_4A2FC20 )
    return ".b32";
  if ( a1 == &off_4A2FCE0 )
    return ".b16";
  if ( a1 == &off_4A2FD40 )
    return ".pred";
  result = "INTERNAL";
  if ( a1 == (__int16 ***)&off_4A2FC80 )
    return "!Special!";
  return result;
}
