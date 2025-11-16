// Function: sub_127CF10
// Address: 0x127cf10
//
__int64 __fastcall sub_127CF10(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int64 v3; // rbx

  result = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 16) = result + 1;
  v3 = *(_QWORD *)(a2 + 72);
  if ( v3 )
  {
    do
    {
      sub_127CD40(a1, v3);
      v3 = *(_QWORD *)(v3 + 16);
    }
    while ( v3 );
    result = *(_QWORD *)(a1 + 16) - 1LL;
  }
  *(_QWORD *)(a1 + 16) = result;
  return result;
}
