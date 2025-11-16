// Function: sub_16E5B30
// Address: 0x16e5b30
//
const char *__fastcall sub_16E5B30(__int128 a1, __int64 a2, _QWORD *a3)
{
  __int64 v5; // [rsp+8h] [rbp-18h] BYREF

  if ( sub_16D2BB0(a1, 0, &v5) )
    return "invalid number";
  *a3 = v5;
  return 0;
}
