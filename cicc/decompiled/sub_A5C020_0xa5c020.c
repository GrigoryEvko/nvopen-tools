// Function: sub_A5C020
// Address: 0xa5c020
//
char __fastcall sub_A5C020(_BYTE *a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v6; // r15
  const __m128i *v7; // rax
  char result; // al

  if ( a3 )
    return sub_A5BCB0((__int64)a1, a2, a3, a4);
  v6 = *(_QWORD *)(a4 + 24);
  v7 = sub_A56340(a4, a2);
  result = sub_A5BC40(a1, a2, (__int64)v7, v6);
  if ( !result )
    return sub_A5BCB0((__int64)a1, a2, a3, a4);
  return result;
}
