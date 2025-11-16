// Function: sub_7FCAA0
// Address: 0x7fcaa0
//
_QWORD *__fastcall sub_7FCAA0(__int64 a1, __m128i *a2, _QWORD *a3)
{
  _QWORD *v4; // r15
  __int64 v5; // rsi
  __int64 v6; // rbx
  _BYTE *v7; // rsi
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r12
  __int64 v12; // r12
  _QWORD *v13; // rax
  bool v14; // zf
  __int64 v15; // r12
  int v16; // edi
  __int64 v18; // [rsp+0h] [rbp-50h]
  __m128i *v19; // [rsp+8h] [rbp-48h]
  int v20; // [rsp+14h] [rbp-3Ch] BYREF
  __int64 v21[7]; // [rsp+18h] [rbp-38h] BYREF

  v19 = a2;
  v18 = sub_7E3E50(a1);
  v21[0] = (__int64)sub_724DC0();
  if ( !a2 )
    v19 = sub_7F7020(a1);
  sub_7296C0(&v20);
  v4 = sub_724D50(10);
  if ( a3 )
  {
    v5 = v21[0];
    LOWORD(v6) = 0;
    do
    {
      while ( 1 )
      {
        LOWORD(v6) = v6 + 1;
        sub_72D510(a3[3], v5, 1);
        v7 = (_BYTE *)sub_7E1DC0();
        sub_70FEE0(v21[0], (__int64)v7, v8, v9, v10);
        v11 = a3[4];
        *(_QWORD *)(v21[0] + 192) = sub_7E1340() * v11;
        *(_BYTE *)(a3[3] + 88LL) |= 4u;
        v12 = sub_724E50(v21, v7);
        v13 = sub_724DC0();
        v14 = v4[22] == 0;
        v21[0] = (__int64)v13;
        v5 = (__int64)v13;
        if ( v14 )
          break;
        *(_QWORD *)(v4[23] + 120LL) = v12;
        v4[23] = v12;
        a3 = (_QWORD *)*a3;
        if ( !a3 )
          goto LABEL_8;
      }
      v4[22] = v12;
      v4[23] = v12;
      a3 = (_QWORD *)*a3;
    }
    while ( a3 );
LABEL_8:
    v6 = (unsigned __int16)v6;
  }
  else
  {
    v6 = 0;
  }
  v15 = v19[7].m128i_i64[1];
  *(_QWORD *)(v15 + 176) = v6;
  sub_8D6090(v15);
  v4[16] = v15;
  v19[11].m128i_i64[1] = (__int64)v4;
  v19[11].m128i_i8[1] = 1;
  v19[10].m128i_i8[14] = *(_BYTE *)(v18 + 174) & 1 | v19[10].m128i_i8[14] & 0xFE;
  v16 = v20;
  v19[8].m128i_i8[8] = *(_BYTE *)(v18 + 136);
  v19[15].m128i_i64[0] = *(_QWORD *)(v18 + 240);
  sub_729730(v16);
  return sub_724E30((__int64)v21);
}
