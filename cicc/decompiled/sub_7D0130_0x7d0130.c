// Function: sub_7D0130
// Address: 0x7d0130
//
__int64 __fastcall sub_7D0130(_QWORD *a1, unsigned __int8 a2, int a3, __int64 a4)
{
  __int64 v6; // r15
  _QWORD *i; // rax
  int v8; // eax
  __m128i *v9; // rax
  __m128i *v10; // r12
  __int64 v12; // rdx
  __int16 v13; // ax
  __m128i *v14; // rax
  __int8 v15; // al
  __m128i *v16; // rax
  char v17; // al
  __int64 v18; // rax
  __int64 v19; // [rsp+8h] [rbp-38h]
  __int64 v20; // [rsp+8h] [rbp-38h]

  v6 = sub_87EBB0(a2, *(_QWORD *)a4);
  for ( i = a1; *((_BYTE *)i + 140) == 12; i = (_QWORD *)i[20] )
    ;
  v8 = *(_DWORD *)(*(_QWORD *)(*i + 96LL) + 96LL);
  *(_BYTE *)(v6 + 84) |= 2u;
  *(_DWORD *)(v6 + 40) = v8;
  if ( a2 == 9 )
  {
    v16 = sub_735FB0(dword_4D03B80, 1, -1);
    *(_QWORD *)(v6 + 88) = v16;
    v10 = v16;
    if ( !v16 )
      goto LABEL_10;
    goto LABEL_8;
  }
  if ( a2 > 9u )
  {
    if ( a2 == 19 )
    {
      v12 = *(_QWORD *)(v6 + 88);
      v13 = *(_WORD *)(v12 + 264);
      *(_BYTE *)(v12 + 160) |= 2u;
      v19 = v12;
      *(_WORD *)(v12 + 264) = v13 & 0x3F00 | 9;
      v14 = (__m128i *)sub_727340();
      v14[7].m128i_i8[8] = 1;
      v10 = v14;
      v14[10].m128i_i64[1] = v19;
      *(_QWORD *)(v19 + 104) = v14;
      if ( dword_4F07590 )
        sub_7344C0((__int64)v14, 0);
      goto LABEL_8;
    }
    goto LABEL_32;
  }
  if ( a2 == 2 )
  {
    v10 = (__m128i *)sub_724D80(12);
    v17 = *(_BYTE *)(a4 + 16);
    if ( (v17 & 0x10) != 0 )
    {
      sub_7249B0((__int64)v10, 3);
      v10[11].m128i_i64[1] = *(_QWORD *)(a4 + 56);
      *(_BYTE *)(v6 + 84) |= 1u;
    }
    else if ( (v17 & 8) != 0 )
    {
      sub_7249B0((__int64)v10, 3);
      if ( (*(_BYTE *)(a4 + 16) & 1) != 0 )
        v10[11].m128i_i8[1] |= 1u;
      v10[12].m128i_i8[8] = *(_BYTE *)(a4 + 56);
      *(_BYTE *)(v6 + 84) |= 1u;
    }
    else if ( (v17 & 0x20) != 0 && (v18 = *(_QWORD *)(a4 + 56)) != 0 )
    {
      if ( (unsigned __int8)(*(_BYTE *)(v18 + 140) - 9) <= 2u && *(_QWORD *)(*(_QWORD *)(v18 + 168) + 256LL) )
        v18 = *(_QWORD *)(*(_QWORD *)(v18 + 168) + 256LL);
      v20 = v18;
      sub_7249B0((__int64)v10, 13);
      v10[11].m128i_i64[1] = v20;
      v10[12].m128i_i8[0] = ((*(_BYTE *)(a4 + 16) & 1) == 0) | v10[12].m128i_i8[0] & 0xFE;
    }
    else
    {
      sub_7249B0((__int64)v10, 2);
    }
    *(_QWORD *)(v6 + 88) = v10;
    v10[8].m128i_i64[0] = dword_4D03B80;
    goto LABEL_8;
  }
  if ( a2 != 3 )
LABEL_32:
    sub_721090();
  v9 = (__m128i *)sub_7259C0(14);
  v9[10].m128i_i8[0] = 1;
  v10 = v9;
  sub_8D6090(v9);
  *(_QWORD *)(v6 + 88) = v10;
LABEL_8:
  sub_877F50(v10, v6, 0xFFFFFFFFLL);
  if ( (a3 & 0x800000) != 0 )
  {
    v15 = v10[5].m128i_i8[10] | 0x10;
    v10[5].m128i_i8[10] = v15;
    v10[5].m128i_i8[10] = (32 * (*(_BYTE *)(a4 + 16) & 1)) | v15 & 0xDF;
  }
LABEL_10:
  if ( !*(_QWORD *)(a1[21] + 152LL) )
    sub_72AD80(a1);
  sub_877E20(v6, v10, a1);
  return v6;
}
