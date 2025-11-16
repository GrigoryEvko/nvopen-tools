// Function: sub_7EB890
// Address: 0x7eb890
//
__m128i *__fastcall sub_7EB890(__int64 a1, __int64 a2)
{
  __int64 v2; // r12
  __m128i *v3; // r13
  __m128i *v5; // r13
  __m128i *v6; // rbx
  __m128i *v7; // rax
  __int64 v8; // rax
  __m128i *v9; // rax
  int v10[9]; // [rsp+Ch] [rbp-24h] BYREF

  v2 = a1;
  v3 = *(__m128i **)(a1 + 160);
  if ( !v3 )
  {
    v5 = *(__m128i **)(a1 + 128);
    if ( (_DWORD)a2 )
    {
      a2 = 1;
      v5 = sub_73C570(*(const __m128i **)(a1 + 128), 1);
    }
    if ( (*(_BYTE *)(a1 - 8) & 1) != 0 )
    {
      if ( !unk_4D03EB8 )
      {
        v6 = (__m128i *)a1;
        goto LABEL_13;
      }
    }
    else if ( !unk_4D03EB8 )
    {
      v3 = sub_7E9300((__int64)v5, 1);
      v9 = sub_7401F0(a1);
      sub_7333B0((__int64)v3, (_BYTE *)qword_4F04C50, 1, (__int64)v9, 0);
LABEL_20:
      *(_QWORD *)(v2 + 160) = v3;
      goto LABEL_7;
    }
    sub_7296C0(v10);
    a2 = 0;
    v6 = sub_740190(a1, 0, 0xAu);
LABEL_13:
    v7 = sub_7E9240((__int64)v5);
    v7[11].m128i_i8[1] = 1;
    v3 = v7;
    v7[11].m128i_i64[1] = (__int64)v6;
    if ( v6[10].m128i_i8[13] == 7
      && (v6[12].m128i_i8[0] & 2) != 0
      && ((v8 = v6[12].m128i_i64[1]) == 0 || (*(_BYTE *)(v8 + 198) & 0x10) != 0) )
    {
      a2 = 1;
      sub_7E1230(v3, 1, 1, 1);
    }
    else if ( (unsigned int)sub_7E0E30() || (v6[10].m128i_i8[8] & 0x10) != 0 )
    {
      a2 = 0;
      sub_7E1230(v3, 0, 1, 1);
      sub_7E1270((__int64)v6);
    }
    sub_7EB800((__int64)v6, (__m128i *)a2);
    if ( unk_4D03EB8 )
    {
      sub_729730(v10[0]);
      *(_QWORD *)(a1 + 160) = v3;
    }
    v2 = (__int64)v6;
    goto LABEL_20;
  }
  if ( (*(_BYTE *)(a1 - 8) & 1) != 0
    && (v3[9].m128i_i8[12] & 1) == 0
    && ((unsigned int)sub_7E0E30() || (*(_BYTE *)(a1 + 168) & 0x10) != 0) )
  {
    sub_7E1230(v3, 0, 0, 0);
    sub_7E1270(a1);
  }
LABEL_7:
  sub_760760((__int64)v3, 7, v2, 0);
  return v3;
}
