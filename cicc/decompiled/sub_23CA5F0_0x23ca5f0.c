// Function: sub_23CA5F0
// Address: 0x23ca5f0
//
unsigned __int64 *__fastcall sub_23CA5F0(unsigned __int64 *a1, char **a2, __int64 *a3, __int64 *a4)
{
  _QWORD *v6; // rax
  unsigned __int64 v7; // r12

  v6 = (_QWORD *)sub_22077B0(0x18u);
  v7 = (unsigned __int64)v6;
  if ( v6 )
  {
    *v6 = 0;
    v6[1] = 0;
    v6[2] = 0x2800000000LL;
    if ( !(unsigned __int8)sub_23CA1B0((__int64)v6, a2, a3, a4) )
    {
      *a1 = 0;
      sub_23C6FB0(v7);
      j_j___libc_free_0(v7);
      return a1;
    }
  }
  else if ( !(unsigned __int8)sub_23CA1B0(0, a2, a3, a4) )
  {
    *a1 = 0;
    return a1;
  }
  *a1 = v7;
  return a1;
}
