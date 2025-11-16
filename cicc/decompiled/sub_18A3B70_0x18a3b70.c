// Function: sub_18A3B70
// Address: 0x18a3b70
//
__int64 __fastcall sub_18A3B70(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // r12
  __int64 i; // rdi
  __int64 v6; // r12
  __int64 k; // r15
  __int64 j; // [rsp+8h] [rbp-38h]

  v3 = 0;
  v4 = a1 + 40;
  for ( i = *(_QWORD *)(a1 + 56); v4 != i; i = sub_220EF30(i) )
    v3 += *(_QWORD *)(i + 40);
  v6 = *(_QWORD *)(a1 + 104);
  for ( j = a1 + 88; j != v6; v6 = sub_220EF30(v6) )
  {
    for ( k = *(_QWORD *)(v6 + 64); v6 + 48 != k; k = sub_220EF30(k) )
    {
      if ( sub_1441CD0(a2, *(_QWORD *)(k + 80)) )
        v3 += sub_18A3B70(k + 64, a2);
    }
  }
  return v3;
}
