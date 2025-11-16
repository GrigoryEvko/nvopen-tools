// Function: sub_F45F60
// Address: 0xf45f60
//
__int64 __fastcall sub_F45F60(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 result; // rax
  __int64 v5; // r12
  _QWORD *v6; // r10
  _QWORD *v7; // rdx
  _QWORD *v8; // r13
  __int64 v9; // rdi
  __int64 v10; // [rsp+8h] [rbp-68h]
  __int64 v11; // [rsp+10h] [rbp-60h]
  __int64 i; // [rsp+18h] [rbp-58h]
  _QWORD *v13; // [rsp+20h] [rbp-50h]
  __int64 v14; // [rsp+28h] [rbp-48h]
  _BYTE v15[56]; // [rsp+38h] [rbp-38h] BYREF

  result = a1 + 8 * a2;
  v10 = result;
  if ( a1 != result )
  {
    v11 = a1;
    do
    {
      v5 = *(_QWORD *)(*(_QWORD *)v11 + 56LL);
      for ( i = *(_QWORD *)v11 + 48LL; i != v5; v5 = *(_QWORD *)(v5 + 8) )
      {
        if ( !v5 )
          BUG();
        v9 = *(_QWORD *)(v5 + 40);
        if ( v9 )
        {
          v6 = (_QWORD *)sub_B14240(v9);
          v8 = v7;
        }
        else
        {
          v8 = &qword_4F81430[1];
          v6 = &qword_4F81430[1];
        }
        v13 = v6;
        v14 = sub_B43CA0(v5 - 24);
        sub_FC75A0(v15, a3, 3, 0, 0, 0);
        sub_FCD310(v15, v14, v13, v8);
        sub_FC7680(v15);
        sub_FC75A0(v15, a3, 3, 0, 0, 0);
        sub_FCD280(v15, v5 - 24);
        sub_FC7680(v15);
      }
      v11 += 8;
      result = v11;
    }
    while ( v10 != v11 );
  }
  return result;
}
