// Function: sub_16D2BB0
// Address: 0x16d2bb0
//
char __fastcall sub_16D2BB0(__int128 a1, unsigned int a2, __int64 *a3)
{
  char result; // al
  __int128 v4; // [rsp+0h] [rbp-10h] BYREF

  v4 = a1;
  result = sub_16D2AE0((__m128i *)&v4, a2, a3);
  if ( !result )
    return *((_QWORD *)&v4 + 1) != 0;
  return result;
}
