// Function: sub_B14600
// Address: 0xb14600
//
__int64 __fastcall sub_B14600(__int64 a1, __int64 a2, __int64 a3, char a4, char a5)
{
  __int64 v5; // r12
  __int64 v6; // r13
  __int64 *v7; // r14
  __int64 result; // rax
  __int64 v9; // r15
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rcx

  v5 = a2 + 8;
  v6 = *(_QWORD *)(a2 + 16);
  if ( a4 )
    v6 = a3;
  v7 = (__int64 *)(a1 + 8);
  if ( a5 )
    v7 = *(__int64 **)(a1 + 16);
  result = a1 + 8;
  v9 = 0;
  if ( v6 != v5 )
  {
    do
    {
      v10 = sub_B13130(v6);
      *(_QWORD *)(v10 + 16) = a1;
      v11 = *(_QWORD *)v10;
      v12 = *v7;
      *(_QWORD *)(v10 + 8) = v7;
      v12 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v10 = v12 | v11 & 7;
      *(_QWORD *)(v12 + 8) = v10;
      *v7 = v10 | *v7 & 7;
      v6 = *(_QWORD *)(v6 + 8);
      if ( !v9 )
        v9 = v10;
    }
    while ( v5 != v6 );
    if ( a5 )
      return *(_QWORD *)(a1 + 16);
    else
      return v9;
  }
  return result;
}
