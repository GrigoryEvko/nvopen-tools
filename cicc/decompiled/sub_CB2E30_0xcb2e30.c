// Function: sub_CB2E30
// Address: 0xcb2e30
//
const char *__fastcall sub_CB2E30(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  if ( sub_C93CC0(a1, a2, 0, &v6) )
    return "invalid number";
  if ( (unsigned __int64)(v6 + 0x80000000LL) > 0xFFFFFFFF )
    return "out of range number";
  *a4 = v6;
  return 0;
}
