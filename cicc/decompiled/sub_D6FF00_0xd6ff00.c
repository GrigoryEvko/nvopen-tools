// Function: sub_D6FF00
// Address: 0xd6ff00
//
unsigned __int64 __fastcall sub_D6FF00(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 result; // rax
  __int64 v4; // r12
  __int64 v5; // rbx
  __int64 v6; // rsi

  result = 3 * a3;
  v4 = a2 + 24 * a3;
  if ( a2 != v4 )
  {
    v5 = a2;
    do
    {
      v6 = *(_QWORD *)(v5 + 16);
      if ( v6 )
        result = sub_D6D630(a1, v6);
      v5 += 24;
    }
    while ( v5 != v4 );
  }
  return result;
}
