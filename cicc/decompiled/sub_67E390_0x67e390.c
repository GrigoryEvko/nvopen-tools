// Function: sub_67E390
// Address: 0x67e390
//
__int64 __fastcall sub_67E390(__int64 *a1, __int64 *a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 result; // rax

  v3 = a1[1];
  if ( a3 )
  {
    if ( a2[1] == a3 )
      a2[1] = v3;
    else
      *(_QWORD *)(v3 + 8) = *(_QWORD *)(a3 + 8);
    result = *a1;
    *(_QWORD *)(a3 + 8) = *a1;
  }
  else
  {
    *(_QWORD *)(v3 + 8) = *a2;
    result = *a1;
    *a2 = *a1;
  }
  return result;
}
