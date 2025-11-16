// Function: sub_9B4FC0
// Address: 0x9b4fc0
//
__int64 __fastcall sub_9B4FC0(unsigned __int8 *a1, __int64 a2, __int64 a3, int a4, const __m128i *a5)
{
  int v5; // eax
  __int64 v7; // r10

  v5 = *a1;
  if ( (unsigned __int8)v5 <= 0x1Cu || (unsigned int)(v5 - 42) > 0x11 )
    return 0;
  if ( v5 == 58 )
  {
    if ( (a1[1] & 2) == 0 )
      return 0;
  }
  else if ( v5 != 59 && v5 != 42 )
  {
    return 0;
  }
  v7 = *((_QWORD *)a1 - 8);
  if ( a2 == v7 )
  {
    v7 = *((_QWORD *)a1 - 4);
    return sub_9A6530(v7, a3, a5, a4 + 1);
  }
  if ( a2 != *((_QWORD *)a1 - 4) )
    return 0;
  return sub_9A6530(v7, a3, a5, a4 + 1);
}
