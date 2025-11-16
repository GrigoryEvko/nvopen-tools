// Function: sub_2BB9190
// Address: 0x2bb9190
//
void __fastcall sub_2BB9190(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rbx
  __int64 v7; // r11
  __int64 v8; // r10
  __int64 v9; // r12
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // r8
  __int64 v15; // r9
  __int64 v16; // r10
  __int64 v17; // r11
  __int64 v18; // r14
  __int64 v19; // rdx
  __int64 v20; // rax
  __int64 v21; // rcx
  __int64 v22; // r8
  __int64 v23; // r9
  __int64 v25; // [rsp+10h] [rbp-50h]
  __int64 v26; // [rsp+18h] [rbp-48h]
  __int64 v27; // [rsp+18h] [rbp-48h]
  __int64 v28; // [rsp+18h] [rbp-48h]
  __int64 v29; // [rsp+20h] [rbp-40h]
  __int64 v30; // [rsp+20h] [rbp-40h]
  __int64 v31; // [rsp+28h] [rbp-38h]
  __int64 v32; // [rsp+28h] [rbp-38h]
  __int64 v33; // [rsp+28h] [rbp-38h]

  if ( a5 )
  {
    v6 = a4;
    if ( a4 )
    {
      v7 = a1;
      v8 = a2;
      v9 = a5;
      if ( a4 + a5 == 2 )
      {
        v18 = a2;
        v19 = a1;
LABEL_13:
        v33 = v19;
        if ( (unsigned __int8)sub_2B1D420(
                                *(unsigned __int8 **)(*(_QWORD *)v18 + 8LL),
                                *(unsigned __int8 **)(*(_QWORD *)v19 + 8LL),
                                v19,
                                a4,
                                a5,
                                a6) )
          sub_2BB8DF0(v33, v18, v33, v21, v22, v23);
      }
      else
      {
        if ( a5 >= a4 )
          goto LABEL_10;
LABEL_5:
        v26 = v7;
        v29 = v8;
        v31 = v6 / 2;
        v11 = v7 + ((v6 / 2) << 6);
        v12 = sub_2B1D790(v8, a3, v11, a6, a5, a6);
        v16 = v29;
        v17 = v26;
        v18 = v12;
        v30 = (v12 - v29) >> 6;
        while ( 1 )
        {
          v27 = v17;
          v25 = sub_2BB8FF0(v11, v16, v18, v13, v14, v15);
          sub_2BB9190(v27, v11, v25, v31, v30, a6);
          v9 -= v30;
          v6 -= v31;
          if ( !v6 )
            break;
          v19 = v25;
          if ( !v9 )
            break;
          if ( v9 + v6 == 2 )
            goto LABEL_13;
          v8 = v18;
          v7 = v25;
          if ( v9 < v6 )
            goto LABEL_5;
LABEL_10:
          v28 = v8;
          v32 = v7;
          v30 = v9 / 2;
          v18 = v8 + ((v9 / 2) << 6);
          v20 = sub_2B1D820(v7, v8, v18, a6, a5, a6);
          v17 = v32;
          v16 = v28;
          v11 = v20;
          v31 = (v20 - v32) >> 6;
        }
      }
    }
  }
}
