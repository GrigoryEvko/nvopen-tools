// Function: sub_394A2C0
// Address: 0x394a2c0
//
unsigned __int64 **__fastcall sub_394A2C0(unsigned __int64 **a1, __int64 *a2, __int64 *a3)
{
  unsigned __int64 *v4; // rax
  unsigned __int64 *v5; // r12

  v4 = (unsigned __int64 *)sub_22077B0(0x18u);
  v5 = v4;
  if ( !v4 )
  {
    if ( !(unsigned __int8)sub_3949E10(0, a2, a3) )
    {
      *a1 = 0;
      return a1;
    }
LABEL_3:
    *a1 = v5;
    return a1;
  }
  *v4 = 0;
  v4[1] = 0;
  v4[2] = 0;
  if ( (unsigned __int8)sub_3949E10(v4, a2, a3) )
    goto LABEL_3;
  *a1 = 0;
  sub_39479B0(v5);
  j_j___libc_free_0((unsigned __int64)v5);
  return a1;
}
