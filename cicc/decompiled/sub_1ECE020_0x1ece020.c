// Function: sub_1ECE020
// Address: 0x1ece020
//
__int64 __fastcall sub_1ECE020(__int64 *a1, __int64 *a2)
{
  char *v3; // rsi
  __int64 result; // rax

  v3 = (char *)a1[1];
  if ( v3 == (char *)a1[2] )
    return sub_1ECDE70(a1, v3, a2);
  if ( v3 )
  {
    result = *a2;
    *(_QWORD *)v3 = *a2;
    *a2 = 0;
    v3 = (char *)a1[1];
  }
  a1[1] = (__int64)(v3 + 8);
  return result;
}
