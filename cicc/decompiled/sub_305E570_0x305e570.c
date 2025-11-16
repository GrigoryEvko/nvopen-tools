// Function: sub_305E570
// Address: 0x305e570
//
void __fastcall sub_305E570(__int64 a1)
{
  __int64 v1; // rax
  __m128i si128; // xmm0
  __int64 v3; // rax
  __m128i v4; // xmm0
  _QWORD *v5; // rax
  unsigned __int64 v6; // [rsp+8h] [rbp-48h] BYREF
  _QWORD *v7; // [rsp+10h] [rbp-40h] BYREF
  unsigned __int64 v8; // [rsp+18h] [rbp-38h]
  _QWORD v9[6]; // [rsp+20h] [rbp-30h] BYREF

  sub_2FF12A0(a1, &unk_503FCF4, 0);
  sub_2FF12A0(a1, &unk_501EB14, 0);
  sub_2FF12A0(a1, &unk_50208AC, 0);
  sub_2FF12A0(a1, &unk_5022C2C, 0);
  sub_2FF12A0(a1, &unk_502A48C, 0);
  sub_2FF12A0(a1, &unk_502476C, 0);
  if ( sub_2FF12A0(a1, &unk_5021074, 0) )
  {
    v7 = v9;
    v6 = 24;
    v1 = sub_22409D0((__int64)&v7, &v6, 0);
    si128 = _mm_load_si128((const __m128i *)&xmmword_44C9F10);
    v7 = (_QWORD *)v1;
    v9[0] = v6;
    *(_QWORD *)(v1 + 16) = 0x676E696C75646568LL;
    *(__m128i *)v1 = si128;
    v8 = v6;
    *((_BYTE *)v7 + v6) = 0;
    sub_2FF0D00(a1, (__int64)&v7);
    if ( v7 != v9 )
      j_j___libc_free_0((unsigned __int64)v7);
  }
  if ( !(_BYTE)qword_502C488 && (unsigned int)sub_2FF0570(a1) )
  {
    v5 = (_QWORD *)sub_308A6E0();
    sub_2FF0E80(a1, v5, 0);
  }
  sub_2FF12A0(a1, &unk_502624C, 0);
  v7 = v9;
  v6 = 23;
  v3 = sub_22409D0((__int64)&v7, &v6, 0);
  v4 = _mm_load_si128((const __m128i *)&xmmword_44C9F20);
  v7 = (_QWORD *)v3;
  v9[0] = v6;
  *(_DWORD *)(v3 + 16) = 1919904879;
  *(_WORD *)(v3 + 20) = 28265;
  *(_BYTE *)(v3 + 22) = 103;
  *(__m128i *)v3 = v4;
  v8 = v6;
  *((_BYTE *)v7 + v6) = 0;
  sub_2FF0D00(a1, (__int64)&v7);
  if ( v7 != v9 )
    j_j___libc_free_0((unsigned __int64)v7);
}
