// Function: sub_2A61100
// Address: 0x2a61100
//
__int64 __fastcall sub_2A61100(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r13
  __int64 i; // rdi
  __int64 k; // r15
  __int64 j; // [rsp+10h] [rbp-40h]

  v4 = 0;
  for ( i = *(_QWORD *)(a2 + 96); i != a2 + 80; i = sub_220EF30(i) )
    v4 += *(_QWORD *)(i + 40);
  for ( j = *(_QWORD *)(a2 + 144); a2 + 128 != j; j = sub_220EF30(j) )
  {
    for ( k = *(_QWORD *)(j + 64); j + 48 != k; k = sub_220EF30(k) )
    {
      if ( sub_2A60EC0(k + 48, a3, *(_BYTE *)(a1 + 40)) )
        v4 += sub_2A61100(a1, k + 48, a3);
    }
  }
  return v4;
}
