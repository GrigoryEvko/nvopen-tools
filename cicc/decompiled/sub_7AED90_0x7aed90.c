// Function: sub_7AED90
// Address: 0x7aed90
//
unsigned __int64 __fastcall sub_7AED90(__int64 a1)
{
  unsigned __int64 result; // rax
  unsigned __int64 v2; // rdx
  __int64 v3; // rax
  __int64 v4; // rdx

  result = *(_QWORD *)(a1 + 16);
  if ( result )
  {
    v2 = (result >> 3) - 7993 * (result / 0xF9C8);
    v3 = qword_4F08580[v2];
    if ( a1 == v3 )
    {
      result = *(_QWORD *)(a1 + 8);
      qword_4F08580[v2] = result;
    }
    else
    {
      do
      {
        v4 = v3;
        v3 = *(_QWORD *)(v3 + 8);
      }
      while ( a1 != v3 );
      result = *(_QWORD *)(a1 + 8);
      *(_QWORD *)(v4 + 8) = result;
    }
  }
  return result;
}
