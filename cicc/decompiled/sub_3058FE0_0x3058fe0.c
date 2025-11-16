// Function: sub_3058FE0
// Address: 0x3058fe0
//
char *__fastcall sub_3058FE0(__int16 ***a1)
{
  char *result; // rax

  if ( a1 == &off_4A2FB60 )
    return "%f";
  if ( a1 == &off_4A2F980 )
    return "%fd";
  if ( a1 == &off_4A2F8C0 )
    return "%rq";
  if ( a1 == &off_4A2FA40 )
    return "%rd";
  if ( a1 == &off_4A2FC20 )
    return "%r";
  if ( a1 == &off_4A2FCE0 )
    return "%rs";
  if ( a1 == &off_4A2FD40 )
    return "%p";
  result = "INTERNAL";
  if ( a1 == (__int16 ***)&off_4A2FC80 )
    return "!Special!";
  return result;
}
