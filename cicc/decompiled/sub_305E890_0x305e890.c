// Function: sub_305E890
// Address: 0x305e890
//
__int64 __fastcall sub_305E890(__int64 a1)
{
  _QWORD *v1; // rsi
  _QWORD *v2; // rax
  unsigned int v3; // eax
  _QWORD *v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rsi
  _QWORD *v7; // rax
  _QWORD *v9; // rsi
  __int64 v10; // rax
  __m128i si128; // xmm0
  unsigned __int64 v12; // rdx
  __int64 v13; // rax
  _QWORD *v14; // rsi
  unsigned __int64 v15; // [rsp+8h] [rbp-48h] BYREF
  unsigned __int64 v16[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v17[6]; // [rsp+20h] [rbp-30h] BYREF

  v1 = (_QWORD *)sub_2CCA760(qword_502C728);
  sub_2FF0E80(a1, v1, 1u);
  if ( (_BYTE)qword_502C8E8 )
  {
    v9 = (_QWORD *)sub_2D05E00();
    sub_2FF0E80(a1, v9, 0);
    if ( !(_BYTE)qword_502BF48 )
      goto LABEL_3;
  }
  else if ( !(_BYTE)qword_502BF48 )
  {
    goto LABEL_3;
  }
  v15 = 40;
  v16[0] = (unsigned __int64)v17;
  v10 = sub_22409D0((__int64)v16, &v15, 0);
  v16[0] = v10;
  v17[0] = v15;
  *(__m128i *)v10 = _mm_load_si128((const __m128i *)&xmmword_44C9F30);
  si128 = _mm_load_si128((const __m128i *)&xmmword_44C9F40);
  v12 = v16[0];
  *(_QWORD *)(v10 + 32) = 0xA2A2A2A206C6553LL;
  *(__m128i *)(v10 + 16) = si128;
  v16[1] = v15;
  *(_BYTE *)(v12 + v15) = 0;
  v13 = sub_C5F790((__int64)v16, (__int64)&v15);
  v14 = (_QWORD *)sub_B3AAD0(v13, (__int64)v16);
  sub_2FF0E80(a1, v14, 0);
  if ( (_QWORD *)v16[0] != v17 )
    j_j___libc_free_0(v16[0]);
LABEL_3:
  v2 = (_QWORD *)sub_36D03D0();
  sub_2FF0E80(a1, v2, 1u);
  v3 = sub_2FF0570(a1);
  v4 = (_QWORD *)sub_36D8F50(*(_QWORD *)(a1 + 256), v3);
  sub_2FF0E80(a1, v4, 1u);
  v5 = (_QWORD *)sub_3094640();
  sub_2FF0E80(a1, v5, 0);
  v6 = (_QWORD *)sub_3096480(*(_QWORD *)(a1 + 256));
  sub_2FF0E80(a1, v6, 0);
  v7 = (_QWORD *)sub_36F7A90();
  sub_2FF0E80(a1, v7, 1u);
  return 0;
}
