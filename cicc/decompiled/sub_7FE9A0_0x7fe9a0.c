// Function: sub_7FE9A0
// Address: 0x7fe9a0
//
__int64 __fastcall sub_7FE9A0(__int64 a1, __m128i *a2, int a3)
{
  __int64 v3; // r13
  __int64 i; // r12
  char v6; // al
  __int64 v8; // r12
  _QWORD *v9; // rax
  _QWORD *v10; // r15
  __m128i *v11; // r12
  __int8 v12; // al
  __int8 v13; // al
  __int8 v14; // al
  __int8 v15; // al
  __int8 v16; // al
  __int64 v17; // r12
  _QWORD *v18; // rax
  __int64 v19; // [rsp+8h] [rbp-38h]

  v3 = a1;
  for ( i = *(_QWORD *)(a1 + 152); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v6 = *(_BYTE *)(a1 + 174);
  if ( (v6 == 1 || v6 == 2)
    && (((*(_BYTE *)(a1 + 205) & 0x1C) - 8) & 0xF4) == 0
    && (*(_BYTE *)(*(_QWORD *)(*(_QWORD *)(a1 + 40) + 32LL) + 176LL) & 0x10) != 0
    || a2 )
  {
    if ( *(_QWORD *)(*(_QWORD *)(i + 168) + 40LL) )
    {
      v19 = sub_8D71D0(i);
      v8 = sub_7F8700(i);
      v9 = sub_7259C0(7);
      v9[20] = v8;
      v10 = v9;
      *(_BYTE *)(v9[21] + 16LL) = (2 * (dword_4F06968 == 0)) | *(_BYTE *)(v9[21] + 16LL) & 0xFD;
      if ( v19 )
        *(_QWORD *)v9[21] = sub_724EF0(v19);
    }
    else
    {
      v17 = sub_7F8700(i);
      v18 = sub_7259C0(7);
      v18[20] = v17;
      v10 = v18;
      *(_BYTE *)(v18[21] + 16LL) = (2 * (dword_4F06968 == 0)) | *(_BYTE *)(v18[21] + 16LL) & 0xFD;
    }
    v11 = sub_725FD0();
    v11[10].m128i_i8[12] = 2;
    v12 = v11[5].m128i_i8[8];
    v11[12].m128i_i8[1] |= 0x10u;
    v11[9].m128i_i64[1] = (__int64)v10;
    v11[5].m128i_i8[8] = v12 & 0x8F | 0x10;
    sub_7362F0((__int64)v11, 0);
    v13 = *(_BYTE *)(a1 + 198) & 8 | v11[12].m128i_i8[6] & 0xF7;
    v11[12].m128i_i8[6] = v13;
    v14 = *(_BYTE *)(a1 + 198) & 0x10 | v13 & 0xEF;
    v11[12].m128i_i8[6] = v14;
    v15 = *(_BYTE *)(a1 + 198) & 0x20 | v14 & 0xDF;
    v11[12].m128i_i8[6] = v15;
    v16 = *(_BYTE *)(a1 + 198) & 0x40 | v15 & 0xBF;
    v11[12].m128i_i8[6] = v16;
    if ( (*(_QWORD *)(a1 + 192) & 0x240000000LL) != 0 )
      v11[12].m128i_i8[4] |= 2u;
    if ( (*(_BYTE *)(a1 + 193) & 0x10) != 0
      || (*(_QWORD *)(a1 + 200) & 0x8000001000000LL) == 0x8000000000000LL && (*(_BYTE *)(a1 + 192) & 2) == 0
      || (*(_BYTE *)(a1 + 198) & 0x18) != 0 )
    {
      if ( !a3 )
      {
LABEL_18:
        v3 = (__int64)v11;
        sub_7FCF80(a1, (__int64)v11, a2);
        return v3;
      }
    }
    else
    {
      v11[12].m128i_i8[6] = v16 | 8;
      if ( !a3 )
        goto LABEL_18;
    }
    sub_7F6570(a1, **(_QWORD ***)(v11[9].m128i_i64[1] + 168), 0, 1);
    goto LABEL_18;
  }
  return v3;
}
