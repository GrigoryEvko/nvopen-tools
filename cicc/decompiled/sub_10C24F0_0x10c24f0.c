// Function: sub_10C24F0
// Address: 0x10c24f0
//
__int64 __fastcall sub_10C24F0(__int64 *a1, _BYTE *a2, unsigned __int8 *a3)
{
  __int64 v5; // [rsp-19h] [rbp-19h] BYREF

  if ( *a2 <= 0x1Cu )
    return 0;
  LOBYTE(v5) = 0;
  if ( sub_F13D80(a1, (__int64)a2, 1, 0, &v5, 0) )
    return sub_10C2350((__int64)a2, a3);
  else
    return 0;
}
