// Function: sub_16E59E0
// Address: 0x16e59e0
//
const char *__fastcall sub_16E59E0(__int64 a1, __int64 a2, __int64 a3, _DWORD *a4)
{
  unsigned __int64 v6; // [rsp+8h] [rbp-18h] BYREF

  if ( sub_16D2B80(a1, a2, 0, &v6) )
    return "invalid number";
  if ( v6 > 0xFFFFFFFF )
    return "out of range number";
  *a4 = v6;
  return 0;
}
