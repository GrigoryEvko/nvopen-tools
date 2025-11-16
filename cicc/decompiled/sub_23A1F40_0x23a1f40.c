// Function: sub_23A1F40
// Address: 0x23a1f40
//
unsigned __int64 __fastcall sub_23A1F40(unsigned __int64 *a1, unsigned __int64 *a2)
{
  char *v3; // rsi
  unsigned __int64 result; // rax

  v3 = (char *)a1[1];
  if ( v3 == (char *)a1[2] )
    return sub_2353750(a1, v3, a2);
  if ( v3 )
  {
    result = *a2;
    *(_QWORD *)v3 = *a2;
    *a2 = 0;
    v3 = (char *)a1[1];
  }
  a1[1] = (unsigned __int64)(v3 + 8);
  return result;
}
