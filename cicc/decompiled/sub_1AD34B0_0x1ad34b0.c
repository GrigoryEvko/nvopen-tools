// Function: sub_1AD34B0
// Address: 0x1ad34b0
//
unsigned __int64 __fastcall sub_1AD34B0(__int64 *a1, unsigned __int8 **a2)
{
  unsigned __int64 result; // rax
  unsigned __int8 *v4; // rsi

  if ( *a1 )
    result = sub_161E7C0((__int64)a1, *a1);
  v4 = *a2;
  *a1 = (__int64)*a2;
  if ( v4 )
  {
    result = sub_1623210((__int64)a2, v4, (__int64)a1);
    *a2 = 0;
  }
  return result;
}
