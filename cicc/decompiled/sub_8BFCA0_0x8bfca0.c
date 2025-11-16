// Function: sub_8BFCA0
// Address: 0x8bfca0
//
void __fastcall sub_8BFCA0(__m128i *a1, __int64 a2, __int64 a3, __int64 a4)
{
  _QWORD *v5; // r15
  __int64 v6; // r8
  __int64 v7; // r13
  _BYTE *v8; // rbx
  char v9; // al
  __int64 v11; // rcx
  __m128i *i; // r13
  __int64 v13; // rax
  _QWORD *v14; // r8
  __int64 v15; // rcx
  __int64 v16; // r9
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rsi
  char v21; // al
  char v22; // al
  _QWORD *v23; // [rsp+8h] [rbp-48h]
  __int64 v24; // [rsp+10h] [rbp-40h] BYREF
  __int64 v25[7]; // [rsp+18h] [rbp-38h] BYREF

  v5 = *(_QWORD **)(a2 + 96);
  if ( v5 )
  {
    if ( (__m128i *)v5[4] != a1 )
      return;
    v6 = a1[5].m128i_i64[1];
    if ( !v6 )
      return;
    goto LABEL_5;
  }
  v11 = *(_QWORD *)(a2 + 88);
  if ( (((*(_BYTE *)(v11 + 88) & 0x70) - 16) & 0xE0) == 0 )
  {
    for ( i = *(__m128i **)(v11 + 152); i[8].m128i_i8[12] == 12; i = (__m128i *)i[10].m128i_i64[0] )
      ;
    if ( (unsigned int)sub_8B7B10(a1, i, v25, &v24, a3, 0, 1, 0) )
    {
      v13 = sub_880C60();
      *(_BYTE *)(v13 + 80) |= 4u;
      v5 = (_QWORD *)v13;
      *(_QWORD *)(v13 + 32) = a1;
      v14 = (_QWORD *)a1[5].m128i_i64[1];
      v23 = v14;
      *(_QWORD *)v13 = v14[21];
      v15 = v25[0];
      v14[21] = v13;
      sub_890140((__int64)a1, v14, a2, v15, (__int64)v14, v16);
      v5[3] = a2;
      v17 = *(_QWORD *)(a2 + 88);
      *(_QWORD *)(a2 + 96) = v5;
      *(_BYTE *)(v17 + 195) |= 1u;
      v18 = v25[0];
      *(_QWORD *)(v17 + 248) = a4;
      *(_QWORD *)(v17 + 240) = v18;
      v19 = v23[11];
      if ( !v19 || (v23[20] & 1) != 0 || v23[30] )
        v20 = (__int64)(v23 + 23);
      else
        v20 = *(_QWORD *)(v19 + 88) + 184LL;
      v5[14] = sub_624310((__int64)i, v20);
      sub_8CCE20(a2, v23);
      v6 = (__int64)v23;
LABEL_5:
      if ( (*(_BYTE *)(a2 + 81) & 2) != 0 )
      {
        sub_648C10(a2, a2 + 48);
        *(_BYTE *)(*(_QWORD *)(a2 + 88) + 195LL) |= 2u;
        *(_BYTE *)(*(_QWORD *)(a2 + 88) + 195LL) |= 4u;
      }
      else
      {
        v7 = *(_QWORD *)(v6 + 176);
        v8 = *(_BYTE **)(a2 + 88);
        v9 = v8[172];
        if ( *(_BYTE *)(v7 + 172) == 2 )
        {
          if ( v9 != 2 )
          {
            sub_6854B0(0x229u, a2);
            v21 = v8[88];
            v8[200] &= 0xF8u;
            v8[172] = 2;
            v8[88] = v21 & 0x8F | 0x10;
          }
        }
        else if ( v9 == 2 )
        {
          sub_6854B0(0x229u, a2);
          v22 = v8[88];
          v8[172] = 0;
          v8[88] = v22 & 0x8F | 0x20;
        }
        if ( *(char *)(v7 + 192) < 0 )
        {
          if ( (v8[193] & 0x40) != 0 )
            sub_685480(0x1DFu, a2);
          sub_736C90((__int64)v8, 1);
        }
        else if ( (char)v8[192] < 0 )
        {
          sub_6854B0(0x291u, a2);
        }
        sub_8AC530((__int64)v5, (*((_WORD *)v8 + 96) & 0x4001) != 0, 0);
      }
    }
  }
}
