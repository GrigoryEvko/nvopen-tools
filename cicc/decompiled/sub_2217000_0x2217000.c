// Function: sub_2217000
// Address: 0x2217000
//
unsigned __int8 *__fastcall sub_2217000(__int64 a1, unsigned __int8 *a2, unsigned __int8 *a3)
{
  __int64 v4; // rdx

  if ( a2 < a3 )
  {
    do
    {
      v4 = *a2++;
      *(a2 - 1) = *(_DWORD *)(*(_QWORD *)(a1 + 32) + 4 * v4);
    }
    while ( a3 != a2 );
  }
  return a3;
}
