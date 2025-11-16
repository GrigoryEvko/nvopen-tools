// Function: sub_11A05B0
// Address: 0x11a05b0
//
unsigned __int8 *__fastcall sub_11A05B0(__int64 **a1, __int64 a2)
{
  if ( !(_DWORD)a2 )
    return (unsigned __int8 *)sub_ACADE0(a1);
  if ( (_DWORD)a2 == 64 )
    return (unsigned __int8 *)sub_AD6530((__int64)a1, a2);
  if ( (unsigned int)*((unsigned __int8 *)a1 + 8) - 15 <= 1 )
    return 0;
  if ( (_DWORD)a2 == 32 )
    return sub_AD9290((__int64)a1, 1);
  if ( (_DWORD)a2 == 512 )
    return (unsigned __int8 *)sub_AD9500((__int64)a1, 0);
  if ( (_DWORD)a2 != 4 )
    return 0;
  return (unsigned __int8 *)sub_AD9500((__int64)a1, 1);
}
