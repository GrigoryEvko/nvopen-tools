// Function: sub_91D4D0
// Address: 0x91d4d0
//
__int64 __fastcall sub_91D4D0(__int64 a1, __int64 a2)
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
      sub_91D300(a1, v3);
      v3 = *(_QWORD *)(v3 + 16);
    }
    while ( v3 );
    result = *(_QWORD *)(a1 + 16) - 1LL;
  }
  *(_QWORD *)(a1 + 16) = result;
  return result;
}
