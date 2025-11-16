// Function: sub_15A51A0
// Address: 0x15a51a0
//
__int64 __fastcall sub_15A51A0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v5; // rsi
  _QWORD *v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rsi
  __int64 v9; // rsi
  _QWORD v11[5]; // [rsp+8h] [rbp-28h] BYREF

  v5 = *(_QWORD *)(a2 + 16);
  v6 = (_QWORD *)(v5 & 0xFFFFFFFFFFFFFFF8LL);
  if ( (v5 & 4) != 0 )
    v6 = (_QWORD *)*v6;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = v6;
  *(_QWORD *)(a1 + 32) = 0;
  *(_DWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 8) = 0;
  if ( a4 )
  {
    *(_QWORD *)(a1 + 8) = *(_QWORD *)(a4 + 40);
    *(_QWORD *)(a1 + 16) = a4 + 24;
    v7 = *(_QWORD *)(a4 + 48);
    v11[0] = v7;
    if ( v7 )
    {
      sub_1623A60(v11, v7, 2);
      if ( *(_QWORD *)a1 )
        sub_161E7C0(a1);
      v8 = v11[0];
      *(_QWORD *)a1 = v11[0];
      if ( v8 )
        sub_1623210(v11, v8, a1);
    }
  }
  else if ( a3 )
  {
    *(_QWORD *)(a1 + 8) = a3;
    *(_QWORD *)(a1 + 16) = a3 + 40;
  }
  sub_15C7080(v11, a2);
  if ( *(_QWORD *)a1 )
    sub_161E7C0(a1);
  v9 = v11[0];
  *(_QWORD *)a1 = v11[0];
  if ( v9 )
    sub_1623210(v11, v9, a1);
  return a1;
}
