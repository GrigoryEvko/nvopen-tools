// Function: sub_21F6CA0
// Address: 0x21f6ca0
//
__int64 __fastcall sub_21F6CA0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rbx
  _QWORD *v7; // r12
  char v8; // r12
  __int64 v9; // r14
  __int64 v10; // rsi
  _QWORD *v12; // [rsp+8h] [rbp-68h]
  _QWORD *v13; // [rsp+18h] [rbp-58h]
  __int64 v14; // [rsp+20h] [rbp-50h]
  _QWORD *v15; // [rsp+28h] [rbp-48h]
  __int64 v16; // [rsp+30h] [rbp-40h] BYREF
  __int64 v17[7]; // [rsp+38h] [rbp-38h] BYREF

  v6 = a1 + 160;
  v7 = sub_21F6AD0((_QWORD *)(a1 + 160), a2);
  v13 = sub_21F6AD0((_QWORD *)(a1 + 160), *(_QWORD *)(a3 - 24));
  v15 = v13 + 1;
  v14 = v7[3];
  v12 = v7 + 1;
  if ( (_QWORD *)v14 != v7 + 1 )
  {
    do
    {
      v16 = 0;
      v8 = sub_21F29B0(v6, *(_QWORD *)(v14 + 32), &v16, a4);
      v9 = v13[3];
      if ( (_QWORD *)v9 != v15 )
      {
        do
        {
          v10 = *(_QWORD *)(v9 + 32);
          v17[0] = 0;
          if ( (unsigned __int8)sub_21F29B0(v6, v10, v17, a4) )
          {
            if ( v8 && (!v16 || !v17[0] || v16 == v17[0]) )
              return 1;
          }
          else if ( !v8 )
          {
            return 1;
          }
          v9 = sub_220EF30(v9);
        }
        while ( v15 != (_QWORD *)v9 );
      }
      v14 = sub_220EF30(v14);
    }
    while ( v12 != (_QWORD *)v14 );
  }
  return 0;
}
