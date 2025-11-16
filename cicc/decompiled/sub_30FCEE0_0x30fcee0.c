// Function: sub_30FCEE0
// Address: 0x30fcee0
//
__int64 __fastcall sub_30FCEE0(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // r12
  __int64 v3; // rbx
  __int64 i; // r12
  __int64 v5; // rsi
  _QWORD *v6; // rax

  v1 = 0;
  v2 = *(_QWORD *)(a1 + 8);
  v3 = *(_QWORD *)(v2 + 32);
  for ( i = v2 + 24; i != v3; v1 += v6[8] )
  {
    while ( 1 )
    {
      v5 = v3 - 56;
      if ( !v3 )
        v5 = 0;
      if ( !sub_B2FC80(v5) )
        break;
      v3 = *(_QWORD *)(v3 + 8);
      if ( i == v3 )
        return v1;
    }
    v6 = sub_30FCBF0(a1, v5);
    v3 = *(_QWORD *)(v3 + 8);
  }
  return v1;
}
