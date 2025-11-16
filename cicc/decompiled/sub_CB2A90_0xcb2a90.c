// Function: sub_CB2A90
// Address: 0xcb2a90
//
const char *__fastcall sub_CB2A90(_BYTE *a1, __int64 a2, __int64 a3, _BYTE *a4)
{
  unsigned __int16 v5; // ax
  const char *v6; // r8

  v5 = sub_CA7AB0(a1, a2);
  v6 = "invalid boolean";
  if ( HIBYTE(v5) )
  {
    *a4 = v5;
    return 0;
  }
  return v6;
}
