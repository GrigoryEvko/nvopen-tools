// Function: sub_154C150
// Address: 0x154c150
//
__int64 __fastcall sub_154C150(__int64 a1, __int64 a2)
{
  __int64 result; // rax

  result = sub_154BC70(a1);
  if ( result )
  {
    result = *(_QWORD *)(a1 + 24);
    if ( result != a2 )
    {
      if ( result )
        sub_154BF90(*(_QWORD *)(a1 + 32));
      result = *(_QWORD *)(a1 + 32);
      *(_QWORD *)(result + 8) = a2;
      *(_BYTE *)(result + 16) = 0;
      *(_QWORD *)(a1 + 24) = a2;
    }
  }
  return result;
}
