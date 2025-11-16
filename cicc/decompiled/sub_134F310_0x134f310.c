// Function: sub_134F310
// Address: 0x134f310
//
__int64 __fastcall sub_134F310(_QWORD *a1, __int64 a2, __int64 a3, const __m128i *a4, unsigned __int8 a5)
{
  __int64 v5; // rbx
  __m128i v8; // xmm1
  __m128i v9; // xmm0
  __int64 v10; // rax
  __int64 v11; // rsi
  unsigned __int8 v12; // al
  __int64 v14; // [rsp+8h] [rbp-98h]
  __m128i v15; // [rsp+10h] [rbp-90h] BYREF
  __m128i v16; // [rsp+20h] [rbp-80h]
  __int64 v17; // [rsp+30h] [rbp-70h]
  __m128i v18; // [rsp+40h] [rbp-60h]
  __m128i v19; // [rsp+50h] [rbp-50h]
  __int64 v20; // [rsp+60h] [rbp-40h]

  v5 = a2 + 24;
  v14 = *(_QWORD *)(a3 + 32);
  if ( a2 + 24 != v14 )
  {
    while ( 2 )
    {
      v8 = _mm_loadu_si128(a4);
      v9 = _mm_loadu_si128(a4 + 1);
      v10 = a4[2].m128i_i64[0];
      v18 = v8;
      v20 = v10;
      v19 = v9;
      if ( v5 )
      {
        v17 = v10;
        v11 = v5 - 24;
        v15 = v8;
        v16 = v9;
        switch ( *(_BYTE *)(v5 - 8) )
        {
          case 0x1D:
            v12 = sub_134F0E0(a1, v11 & 0xFFFFFFFFFFFFFFFBLL, (__int64)&v15);
            goto LABEL_5;
          case 0x21:
            v12 = sub_134D290((__int64)a1, v11, &v15);
            goto LABEL_5;
          case 0x36:
            v12 = sub_134D040((__int64)a1, v11, &v15, (__int64)a4);
            goto LABEL_5;
          case 0x37:
            v12 = sub_134D0E0((__int64)a1, v11, &v15, (__int64)a4);
            goto LABEL_5;
          case 0x39:
            v12 = sub_134D190((__int64)a1, v11, &v15);
            goto LABEL_5;
          case 0x3A:
            v12 = sub_134D2D0((__int64)a1, v11, &v15);
            goto LABEL_5;
          case 0x3B:
            v12 = sub_134D360((__int64)a1, v11, &v15);
            goto LABEL_5;
          case 0x4A:
            v12 = sub_134D250((__int64)a1, v11, &v15);
            goto LABEL_5;
          case 0x4E:
            if ( (a5 & (unsigned __int8)sub_134F0E0(a1, v11 | 4, (__int64)&v15) & 3) != 0 )
              return 1;
            goto LABEL_6;
          case 0x52:
            v12 = sub_134D1D0((__int64)a1, v11, &v15);
LABEL_5:
            if ( (a5 & v12 & 3) == 0 )
              goto LABEL_6;
            return 1;
          default:
LABEL_6:
            v5 = *(_QWORD *)(v5 + 8);
            if ( v14 == v5 )
              return 0;
            continue;
        }
      }
      break;
    }
    v17 = v10;
    v15 = v8;
    v16 = v9;
    BUG();
  }
  return 0;
}
