// Function: sub_2E42F50
// Address: 0x2e42f50
//
void __fastcall sub_2E42F50(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  __int64 *v3; // r14

  v1 = a1 + 88;
  v2 = *(_QWORD *)(a1 + 88);
  if ( a1 + 88 != v2 )
  {
    do
    {
      if ( !(unsigned __int8)sub_2E3DDD0(a1, *(_QWORD *)(v1 + 8) + 16LL) )
      {
        v3 = *(__int64 **)(v1 + 8);
        sub_2E42D20(a1, (__int64)(v3 + 2), v1);
        v1 = *v3;
        if ( !(unsigned __int8)sub_2E3DDD0(a1, *(_QWORD *)(*v3 + 8) + 16LL) )
          BUG();
      }
      v1 = *(_QWORD *)(v1 + 8);
    }
    while ( v2 != v1 );
  }
}
