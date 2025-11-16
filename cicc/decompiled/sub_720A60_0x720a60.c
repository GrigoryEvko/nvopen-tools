// Function: sub_720A60
// Address: 0x720a60
//
__int64 __fastcall sub_720A60(__int64 *a1, __int64 a2)
{
  char *v2; // rax
  const char *v4; // rdi

  v2 = getenv("USR_INCLUDE");
  v4 = v2;
  if ( !v2 )
    v4 = "/usr/include";
  return sub_720930((__int64)v4, 1, a1, a2);
}
