// Function: sub_CB3030
// Address: 0xcb3030
//
const char *__fastcall sub_CB3030(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  unsigned __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  if ( sub_C93C90(a1, a2, 0, &v6) )
    return "invalid hex32 number";
  if ( v6 > 0xFFFFFFFF )
    return "out of range hex32 number";
  *a4 = v6;
  return 0;
}
