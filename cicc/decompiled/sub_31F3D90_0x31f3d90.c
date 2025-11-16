// Function: sub_31F3D90
// Address: 0x31f3d90
//
const char *__fastcall sub_31F3D90(__int64 a1)
{
  unsigned __int16 v1; // ax
  __int64 v2; // rax

  v1 = sub_AF18C0(a1);
  if ( v1 > 0x39u )
    return 0;
  v2 = 1LL << v1;
  if ( (v2 & 0x880014) != 0 )
    return "<unnamed-tag>";
  if ( (v2 & 0x200000000000000LL) == 0 )
    return 0;
  return "`anonymous namespace'";
}
