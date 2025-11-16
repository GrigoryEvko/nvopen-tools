// Function: sub_CB2D60
// Address: 0xcb2d60
//
const char *__fastcall sub_CB2D60(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  unsigned __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  if ( sub_C93C90(a1, a2, 0, &v6) )
    return "invalid number";
  if ( v6 > 0xFFFFFFFF )
    return "out of range number";
  *a4 = v6;
  return 0;
}
