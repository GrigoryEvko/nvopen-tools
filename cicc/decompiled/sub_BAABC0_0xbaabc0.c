// Function: sub_BAABC0
// Address: 0xbaabc0
//
const char *__fastcall sub_BAABC0(__int64 a1)
{
  __int64 v1; // rax

  v1 = sub_BA91D0(a1, "darwin.target_variant.triple", 0x1Cu);
  if ( v1 )
    return (const char *)sub_B91420(v1);
  else
    return byte_3F871B3;
}
