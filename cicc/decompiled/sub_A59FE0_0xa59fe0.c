// Function: sub_A59FE0
// Address: 0xa59fe0
//
__int64 __fastcall sub_A59FE0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 j; // r13
  __int64 v5; // rdi
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rbx
  __int64 k; // r15
  __int64 i; // [rsp+10h] [rbp-40h]

  sub_A59D20(a1, a2);
  result = *(_QWORD *)(a2 + 80);
  for ( i = result; a2 + 72 != result; i = result )
  {
    if ( !i )
      BUG();
    for ( j = *(_QWORD *)(i + 32); i + 24 != j; j = *(_QWORD *)(j + 8) )
    {
      if ( !j )
        BUG();
      v5 = *(_QWORD *)(j + 40);
      if ( v5 )
      {
        v6 = sub_B14240(v5);
        v8 = v7;
        for ( k = v6; v8 != k; k = *(_QWORD *)(k + 8) )
          sub_A59DB0(a1, k);
      }
      sub_A59EA0(a1, j - 24);
    }
    result = *(_QWORD *)(i + 8);
  }
  return result;
}
