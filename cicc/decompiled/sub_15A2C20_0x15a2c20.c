// Function: sub_15A2C20
// Address: 0x15a2c20
//
__int64 __fastcall sub_15A2C20(__int64 *a1, __int64 a2, unsigned __int8 a3, char a4, double a5, double a6, double a7)
{
  __int64 v8; // rcx

  v8 = a3;
  if ( a4 )
    v8 = a3 | 2u;
  return sub_15A2A30((__int64 *)0xF, a1, a2, v8, 0, a5, a6, a7);
}
