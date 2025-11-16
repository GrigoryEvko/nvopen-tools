// Function: sub_9CC180
// Address: 0x9cc180
//
__int64 __fastcall sub_9CC180(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v4; // r14
  __int64 v5; // rbx
  __int64 v6; // rcx
  __int64 v7; // rax

  result = a1[1];
  v4 = a3 - a2;
  v5 = (a3 - a2) >> 3;
  if ( (unsigned __int64)(v5 + result) > a1[2] )
  {
    sub_C8D290(a1, a1 + 3, v5 + result, 2);
    result = a1[1];
  }
  v6 = *a1 + 2 * result;
  if ( v4 > 0 )
  {
    v7 = 0;
    do
    {
      *(_WORD *)(v6 + 2 * v7) = *(_QWORD *)(a2 + 8 * v7);
      ++v7;
    }
    while ( v5 - v7 > 0 );
    result = a1[1];
  }
  a1[1] = result + v5;
  return result;
}
