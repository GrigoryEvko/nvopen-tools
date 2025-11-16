// Function: sub_16E5AB0
// Address: 0x16e5ab0
//
const char *__fastcall sub_16E5AB0(__int128 a1, __int64 a2, _DWORD *a3)
{
  __int64 v5; // [rsp+8h] [rbp-18h] BYREF

  if ( sub_16D2BB0(a1, 0, &v5) )
    return "invalid number";
  if ( (unsigned __int64)(v5 + 0x80000000LL) > 0xFFFFFFFF )
    return "out of range number";
  *a3 = v5;
  return 0;
}
