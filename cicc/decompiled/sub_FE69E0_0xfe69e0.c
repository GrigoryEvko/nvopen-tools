// Function: sub_FE69E0
// Address: 0xfe69e0
//
void __fastcall sub_FE69E0(_QWORD *a1)
{
  __int64 v1; // r12
  _QWORD *v2; // r13
  __int64 *v3; // r14

  v1 = (__int64)(a1 + 11);
  v2 = (_QWORD *)a1[11];
  if ( a1 + 11 != v2 )
  {
    do
    {
      if ( !(unsigned __int8)sub_FE1320(a1, *(_QWORD *)(v1 + 8) + 16LL) )
      {
        v3 = *(__int64 **)(v1 + 8);
        sub_FE67B0(a1, (__int64)(v3 + 2), v1);
        v1 = *v3;
        if ( !(unsigned __int8)sub_FE1320(a1, *(_QWORD *)(*v3 + 8) + 16LL) )
          BUG();
      }
      v1 = *(_QWORD *)(v1 + 8);
    }
    while ( v2 != (_QWORD *)v1 );
  }
}
