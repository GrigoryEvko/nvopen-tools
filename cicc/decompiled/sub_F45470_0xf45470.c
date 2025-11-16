// Function: sub_F45470
// Address: 0xf45470
//
void __fastcall sub_F45470(__int64 a1, __int64 a2)
{
  __int64 v2; // rcx
  __int64 v3; // r13
  __int64 v4; // rbx
  __int64 v5; // r12
  __int64 v6; // rdx
  __int64 v7; // rax
  __int64 v8; // [rsp-40h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 40);
  if ( v2 )
  {
    v3 = a1 + 72;
    v4 = *(_QWORD *)(a1 + 80);
    if ( a1 + 72 == v4 )
    {
      v5 = 0;
    }
    else
    {
      if ( !v4 )
        BUG();
      while ( 1 )
      {
        v5 = *(_QWORD *)(v4 + 32);
        if ( v5 != v4 + 24 )
          break;
        v4 = *(_QWORD *)(v4 + 8);
        if ( v3 == v4 )
          return;
        if ( !v4 )
          BUG();
      }
    }
    while ( v3 != v4 )
    {
      v6 = v5 - 24;
      if ( !v5 )
        v6 = 0;
      v8 = v2;
      sub_AE8BE0(a2, v2, v6);
      v5 = *(_QWORD *)(v5 + 8);
      v2 = v8;
      while ( 1 )
      {
        v7 = v4 - 24;
        if ( !v4 )
          v7 = 0;
        if ( v5 != v7 + 48 )
          break;
        v4 = *(_QWORD *)(v4 + 8);
        if ( v3 == v4 )
          return;
        if ( !v4 )
          BUG();
        v5 = *(_QWORD *)(v4 + 32);
      }
    }
  }
}
