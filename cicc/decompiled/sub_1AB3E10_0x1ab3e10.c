// Function: sub_1AB3E10
// Address: 0x1ab3e10
//
__int64 __fastcall sub_1AB3E10(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // r15
  __int64 i; // r12
  __int64 v5; // r13
  __int64 v6; // [rsp+0h] [rbp-50h]
  __int64 v7; // [rsp+8h] [rbp-48h]
  _BYTE v8[56]; // [rsp+18h] [rbp-38h] BYREF

  result = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  v6 = result;
  if ( result != *(_QWORD *)a1 )
  {
    v7 = *(_QWORD *)a1;
    do
    {
      v3 = *(_QWORD *)(*(_QWORD *)v7 + 48LL);
      for ( i = *(_QWORD *)v7 + 40LL; i != v3; v3 = *(_QWORD *)(v3 + 8) )
      {
        v5 = 0;
        if ( v3 )
          v5 = v3 - 24;
        sub_1B75040(v8, a2, 3, 0, 0);
        sub_1B79630(v8, v5);
        sub_1B75110(v8);
      }
      v7 += 8;
      result = v7;
    }
    while ( v6 != v7 );
  }
  return result;
}
