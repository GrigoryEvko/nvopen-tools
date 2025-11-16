// Function: sub_879190
// Address: 0x879190
//
__int64 __fastcall sub_879190(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rdx
  __int64 result; // rax

  v2 = *(_QWORD *)(a2 + 88);
  if ( v2 == a1 )
  {
    result = *(_QWORD *)(v2 + 8);
    *(_QWORD *)(a2 + 88) = result;
  }
  else
  {
    do
    {
      v3 = v2;
      v2 = *(_QWORD *)(v2 + 8);
    }
    while ( a1 != v2 );
    result = *(_QWORD *)(v2 + 8);
    *(_QWORD *)(v3 + 8) = result;
  }
  return result;
}
