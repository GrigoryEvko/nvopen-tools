// Function: sub_2635180
// Address: 0x2635180
//
void __fastcall sub_2635180(__int64 a1, __int64 a2, __int64 a3, signed __int64 a4, signed __int64 a5)
{
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // r14
  __int64 v8; // r15
  __int64 v9; // r12
  __int64 v10; // rbx
  __int64 v11; // r13
  __int64 v12; // r9
  __int64 v13; // r10
  __int64 v14; // r11
  __int64 v15; // r14
  __int64 v16; // r15
  __int64 v17; // [rsp-58h] [rbp-58h]
  __int64 v18; // [rsp-50h] [rbp-50h]
  __int64 v19; // [rsp-48h] [rbp-48h]
  __int64 v20; // [rsp-40h] [rbp-40h]
  __int64 v21; // [rsp-30h] [rbp-30h]
  __int64 v22; // [rsp-20h] [rbp-20h]
  __int64 v23; // [rsp-18h] [rbp-18h]
  __int64 v24; // [rsp-10h] [rbp-10h]

  while ( a4 )
  {
    v24 = v8;
    v23 = v7;
    v22 = v6;
    v9 = a5;
    v21 = v5;
    if ( !a5 )
      break;
    v10 = a4;
    if ( a4 + a5 == 2 )
    {
      if ( *(_QWORD *)(a2 + 40) < *(_QWORD *)(a1 + 40) )
        sub_1888690(a1, a2);
      return;
    }
    if ( a4 > a5 )
    {
      v16 = a4 / 2;
      v15 = a1 + 16 * (a4 / 2 + ((a4 + ((unsigned __int64)a4 >> 63)) & 0xFFFFFFFFFFFFFFFELL));
      v11 = sub_261AC10(a2, a3, v15);
      v20 = 0xAAAAAAAAAAAAAAABLL * ((v11 - v12) >> 4);
    }
    else
    {
      v20 = a5 / 2;
      v11 = a2 + 16 * (a5 / 2 + ((a5 + ((unsigned __int64)a5 >> 63)) & 0xFFFFFFFFFFFFFFFELL));
      v15 = sub_261AC80(a1, a2, v11);
      v16 = 0xAAAAAAAAAAAAAAABLL * ((v15 - a1) >> 4);
    }
    v17 = v14;
    v18 = v13;
    v19 = sub_2634FC0(v15, v12, v11);
    sub_2635180(v18, v15, v19, v16, v20);
    a4 = v10 - v16;
    a2 = v11;
    v5 = v21;
    a5 = v9 - v20;
    a3 = v17;
    a1 = v19;
    v6 = v22;
    v7 = v23;
    v8 = v24;
  }
}
