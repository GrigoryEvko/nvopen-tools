// Function: sub_2E79DC0
// Address: 0x2e79dc0
//
__int64 __fastcall sub_2E79DC0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 i; // rbx

  v3 = (__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 5;
  if ( v3 )
  {
    for ( i = 0; i != v3; ++i )
      sub_2E79D70(a1, i, a2, a3);
  }
  return 0;
}
