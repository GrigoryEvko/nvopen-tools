// Function: sub_2FE6050
// Address: 0x2fe6050
//
__int64 __fastcall sub_2FE6050(int a1, unsigned int a2, unsigned __int16 a3)
{
  __int64 v3; // rbp
  __int64 result; // rax
  __int64 v5; // rdx
  __int64 v6; // [rsp-58h] [rbp-58h] BYREF
  __int64 v7; // [rsp-50h] [rbp-50h]
  __int64 v8; // [rsp-48h] [rbp-48h]
  __int64 v9; // [rsp-40h] [rbp-40h]
  __int64 v10; // [rsp-38h] [rbp-38h]
  __int64 v11; // [rsp-30h] [rbp-30h]
  __int64 v12; // [rsp-28h] [rbp-28h]
  __int64 v13; // [rsp-20h] [rbp-20h]
  __int64 v14; // [rsp-18h] [rbp-18h]
  __int64 v15; // [rsp-10h] [rbp-10h]
  __int64 v16; // [rsp-8h] [rbp-8h]

  result = 729;
  if ( (unsigned __int16)(a3 - 2) <= 7u )
  {
    v5 = *(_QWORD *)&byte_444C4A0[16 * a3 - 16] >> 3;
    v16 = v3;
    switch ( a1 )
    {
      case 340:
        v6 = 0x25D0000025CLL;
        v7 = 0x25F0000025ELL;
        v8 = 0x26100000260LL;
        v9 = 0x26300000262LL;
        v10 = 0x26500000264LL;
        v11 = 0x26700000266LL;
        v12 = 0x26900000268LL;
        v13 = 0x26B0000026ALL;
        v14 = 0x26D0000026CLL;
        v15 = 0x26F0000026ELL;
        result = sub_2FE5F90((__int64)&v6, a2, v5);
        break;
      case 341:
      case 344:
      case 345:
        result = 729;
        break;
      case 342:
        v6 = 0x27100000270LL;
        v7 = 0x27300000272LL;
        v8 = 0x27500000274LL;
        v9 = 0x27700000276LL;
        v10 = 0x27900000278LL;
        v11 = 0x27B0000027ALL;
        v12 = 0x27D0000027CLL;
        v13 = 0x27F0000027ELL;
        v14 = 0x28100000280LL;
        v15 = 0x28300000282LL;
        result = sub_2FE5F90((__int64)&v6, a2, v5);
        break;
      case 343:
        v6 = 0x28500000284LL;
        v7 = 0x28700000286LL;
        v8 = 0x28900000288LL;
        v9 = 0x28B0000028ALL;
        v10 = 0x28D0000028CLL;
        v11 = 0x28F0000028ELL;
        v12 = 0x29100000290LL;
        v13 = 0x29300000292LL;
        v14 = 0x29500000294LL;
        v15 = 0x29700000296LL;
        result = sub_2FE5F90((__int64)&v6, a2, v5);
        break;
      case 346:
        v6 = 0x2AD000002ACLL;
        v7 = 0x2AF000002AELL;
        v8 = 0x2B1000002B0LL;
        v9 = 0x2B3000002B2LL;
        v10 = 0x2B5000002B4LL;
        v11 = 0x2B7000002B6LL;
        v12 = 0x2B9000002B8LL;
        v13 = 0x2BB000002BALL;
        v14 = 0x2BD000002BCLL;
        v15 = 0x2BF000002BELL;
        result = sub_2FE5F90((__int64)&v6, a2, v5);
        break;
      case 347:
        v6 = 0x29900000298LL;
        v7 = 0x29B0000029ALL;
        v8 = 0x29D0000029CLL;
        v9 = 0x29F0000029ELL;
        v10 = 0x2A1000002A0LL;
        v11 = 0x2A3000002A2LL;
        v12 = 0x2A5000002A4LL;
        v13 = 0x2A7000002A6LL;
        v14 = 0x2A9000002A8LL;
        v15 = 0x2AB000002AALL;
        result = sub_2FE5F90((__int64)&v6, a2, v5);
        break;
      case 348:
        v6 = 0x2C1000002C0LL;
        v7 = 0x2C3000002C2LL;
        v8 = 0x2C5000002C4LL;
        v9 = 0x2C7000002C6LL;
        v10 = 0x2C9000002C8LL;
        v11 = 0x2CB000002CALL;
        v12 = 0x2CD000002CCLL;
        v13 = 0x2CF000002CELL;
        v14 = 0x2D1000002D0LL;
        v15 = 0x2D3000002D2LL;
        result = sub_2FE5F90((__int64)&v6, a2, v5);
        break;
      default:
        result = 729;
        break;
    }
  }
  return result;
}
