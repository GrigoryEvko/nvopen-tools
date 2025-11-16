// Function: sub_FD9C40
// Address: 0xfd9c40
//
char __fastcall sub_FD9C40(__int64 **a1, __int64 a2, __m128i a3)
{
  __int64 v3; // r12
  __int64 v4; // rbx
  unsigned __int8 *v5; // rsi
  char result; // al

  v3 = a2 + 48;
  v4 = *(_QWORD *)(a2 + 56);
  if ( v4 != a2 + 48 )
  {
    do
    {
      v5 = (unsigned __int8 *)(v4 - 24);
      if ( !v4 )
        v5 = 0;
      result = sub_FD98A0(a1, v5, a3);
      v4 = *(_QWORD *)(v4 + 8);
    }
    while ( v3 != v4 );
  }
  return result;
}
