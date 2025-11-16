// Function: sub_AE8D20
// Address: 0xae8d20
//
__int64 __fastcall sub_AE8D20(__int64 a1, __int64 a2)
{
  __int64 v3; // r13
  int v4; // eax
  int v5; // r13d
  __int64 v6; // rsi
  __int64 result; // rax
  __int64 v8; // rbx
  __int64 v9; // rsi
  __int64 j; // r15
  __int64 v11; // rdx
  __int64 v12; // [rsp+8h] [rbp-68h]
  __int64 i; // [rsp+18h] [rbp-58h]
  __int64 v14; // [rsp+20h] [rbp-50h] BYREF
  int v15; // [rsp+28h] [rbp-48h]
  __int64 v16; // [rsp+30h] [rbp-40h] BYREF
  int v17; // [rsp+38h] [rbp-38h]

  v3 = sub_BA8DC0(a2, "llvm.dbg.cu", 11);
  v4 = 0;
  if ( v3 )
    v4 = sub_B91A00(v3);
  v17 = v4;
  v16 = v3;
  sub_BA95A0(&v16);
  v14 = v3;
  v15 = 0;
  sub_BA95A0(&v14);
  v5 = v17;
  v17 = v15;
  v16 = v14;
  while ( v17 != v5 )
  {
    v6 = sub_BA9580(&v16);
    sub_AE8620(a1, v6);
    ++v17;
    sub_BA95A0(&v16);
  }
  result = a2 + 24;
  v12 = *(_QWORD *)(a2 + 32);
  if ( v12 != a2 + 24 )
  {
    do
    {
      v8 = v12 - 56;
      if ( !v12 )
        v8 = 0;
      v9 = sub_B92180(v8);
      if ( v9 )
        sub_AE8440(a1, v9);
      for ( i = *(_QWORD *)(v8 + 80); v8 + 72 != i; i = *(_QWORD *)(i + 8) )
      {
        if ( !i )
          BUG();
        for ( j = *(_QWORD *)(i + 32); i + 24 != j; j = *(_QWORD *)(j + 8) )
        {
          v11 = j - 24;
          if ( !j )
            v11 = 0;
          sub_AE8BE0(a1, a2, v11);
        }
      }
      result = *(_QWORD *)(v12 + 8);
      v12 = result;
    }
    while ( a2 + 24 != result );
  }
  return result;
}
