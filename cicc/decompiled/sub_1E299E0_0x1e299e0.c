// Function: sub_1E299E0
// Address: 0x1e299e0
//
__int64 *__fastcall sub_1E299E0(__int64 *a1, __int64 a2)
{
  __int64 v4; // rax
  __int64 v5; // rdi
  __int64 v6; // rsi
  unsigned __int8 *v7; // rsi
  __int64 v9; // rax
  __int64 v10; // rdi
  __int64 v11; // rsi
  _QWORD v12[5]; // [rsp+8h] [rbp-28h] BYREF

  v4 = sub_1E29990(a2);
  if ( v4 )
  {
    v5 = *(_QWORD *)(v4 + 40);
    if ( v5 )
    {
      v6 = *(_QWORD *)(sub_157EBA0(v5) + 48);
      v12[0] = v6;
      if ( v6 )
      {
        sub_1623A60((__int64)v12, v6, 2);
        v7 = (unsigned __int8 *)v12[0];
        if ( v12[0] )
        {
          *a1 = v12[0];
          sub_1623210((__int64)v12, v7, (__int64)a1);
          return a1;
        }
      }
    }
  }
  v9 = **(_QWORD **)(a2 + 32);
  if ( v9 && (v10 = *(_QWORD *)(v9 + 40)) != 0 )
  {
    v11 = *(_QWORD *)(sub_157EBA0(v10) + 48);
    *a1 = v11;
    if ( !v11 )
      return a1;
    sub_1623A60((__int64)a1, v11, 2);
    return a1;
  }
  else
  {
    *a1 = 0;
    return a1;
  }
}
