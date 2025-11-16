// Function: sub_1DE2420
// Address: 0x1de2420
//
void __fastcall sub_1DE2420(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // r13
  __int64 *v3; // r14
  __int64 v4; // r12

  v1 = a1 + 88;
  v2 = *(_QWORD *)(a1 + 88);
  if ( a1 + 88 != v2 )
  {
    do
    {
      while ( (unsigned __int8)sub_1DDED90(a1, *(_QWORD *)(v1 + 8) + 16LL) )
      {
        v1 = *(_QWORD *)(v1 + 8);
        if ( v2 == v1 )
          return;
      }
      v3 = *(__int64 **)(v1 + 8);
      sub_1DE2200(a1, (__int64)(v3 + 2), v1);
      v4 = *v3;
      sub_1DDED90(a1, *(_QWORD *)(*v3 + 8) + 16LL);
      v1 = *(_QWORD *)(v4 + 8);
    }
    while ( v2 != v1 );
  }
}
