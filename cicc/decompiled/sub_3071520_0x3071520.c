// Function: sub_3071520
// Address: 0x3071520
//
__int64 __fastcall sub_3071520(__int64 a1)
{
  _QWORD *v1; // rax
  __m128i *v2; // rsi
  __int64 v3; // rbx
  _QWORD *v4; // rax
  _QWORD *v5; // rsi
  _QWORD *v6; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // rsi
  _QWORD *v9; // rax
  __int64 result; // rax
  _QWORD *v11; // rax
  _QWORD *v12; // rax
  _QWORD *v13; // rax
  _QWORD *v14; // rax
  _QWORD *v15; // rsi
  _QWORD *v16; // rsi
  _QWORD *v17; // rax
  _QWORD *v18; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rax
  _QWORD *v21; // rax
  __m128i v22; // [rsp+0h] [rbp-40h] BYREF
  __int64 (__fastcall *v23)(__m128i *, __int64, int); // [rsp+10h] [rbp-30h]
  _QWORD *(__fastcall *v24)(__int64, __int64, __int64, unsigned __int64 *); // [rsp+18h] [rbp-28h]

  sub_2FF3AB0(a1, (__int64)&unk_503FCFC, 0, 0);
  sub_2FF3AB0(a1, (__int64)&unk_50201DC, 0, 0);
  sub_2FF3AB0(a1, (__int64)&unk_501F390, 0, 0);
  sub_2FF3AB0(a1, (__int64)&unk_5026411, 0, 0);
  sub_2FF3AB0(a1, (__int64)&unk_5040114, 0, 0);
  sub_2FF3AB0(a1, (__int64)&unk_5021D24, 0, 0);
  sub_2FF3AB0(a1, (__int64)&unk_5022FAC, 0, 0);
  sub_2FF3AB0(a1, (__int64)&unk_503B12C, 0, 0);
  sub_2FF3AB0(a1, (__int64)&unk_50226EC, 0, 0);
  sub_2FF3AB0(a1, (__int64)&unk_503FF48, 0, 0);
  sub_2FF3AB0(a1, (__int64)&unk_503FF3C, 0, 0);
  sub_2FF3AB0(a1, (__int64)&unk_503BDC4, 0, 0);
  v1 = (_QWORD *)sub_36CEF90();
  sub_2FF0E80(a1, v1, 0);
  v24 = sub_3060820;
  v23 = (__int64 (__fastcall *)(__m128i *, __int64, int))sub_305B7B0;
  v2 = sub_CF6D20(&v22, 1);
  sub_2FF0E80(a1, v2, 0);
  if ( v23 )
    v23(&v22, (__int64)&v22, 3);
  v3 = *(_QWORD *)(a1 + 256);
  if ( (unsigned int)sub_2FF0570(a1) )
  {
    v16 = (_QWORD *)sub_36D4FA0();
    sub_2FF0E80(a1, v16, 0);
  }
  v4 = (_QWORD *)sub_36D0B80();
  sub_2FF0E80(a1, v4, 0);
  v5 = (_QWORD *)sub_31CF610();
  sub_2FF0E80(a1, v5, 0);
  if ( (unsigned int)sub_2FF0570(a1) )
  {
    v11 = (_QWORD *)sub_2952040(0);
    sub_2FF0E80(a1, v11, 0);
    v12 = (_QWORD *)sub_297E780();
    sub_2FF0E80(a1, v12, 0);
    v13 = (_QWORD *)sub_F11260();
    sub_2FF0E80(a1, v13, 0);
    sub_305CEC0(a1);
    v14 = (_QWORD *)sub_28C1B50();
    sub_2FF0E80(a1, v14, 0);
    v15 = (_QWORD *)sub_277BAB0(0);
    sub_2FF0E80(a1, v15, 0);
  }
  v6 = (_QWORD *)sub_2A85BB0(1);
  sub_2FF0E80(a1, v6, 0);
  if ( (unsigned int)sub_2FF0570(a1) && (_BYTE)qword_502B908 )
  {
    v20 = (_QWORD *)sub_2CB23A0(*(_DWORD *)(v3 + 1628) / 0xAu);
    sub_2FF0E80(a1, v20, 0);
    v21 = (_QWORD *)sub_2752860();
    sub_2FF0E80(a1, v21, 0);
  }
  v7 = (_QWORD *)sub_2D45FB0();
  sub_2FF0E80(a1, v7, 0);
  v8 = (_QWORD *)sub_36D30E0();
  sub_2FF0E80(a1, v8, 0);
  if ( !(_BYTE)qword_502BBC8 )
  {
    v19 = (_QWORD *)sub_36CD4E0();
    sub_2FF0E80(a1, v19, 0);
  }
  v9 = (_QWORD *)sub_36CD270((unsigned __int8)byte_502BD88 ^ 1u, (unsigned __int8)byte_502BCA8 ^ 1u);
  sub_2FF0E80(a1, v9, 0);
  sub_2FF1680(a1);
  result = sub_2FF0570(a1);
  if ( (_DWORD)result )
  {
    result = sub_305CEC0(a1);
    if ( !(_BYTE)qword_502CC68 )
    {
      v18 = (_QWORD *)sub_2A8DB50();
      result = sub_2FF0E80(a1, v18, 0);
    }
  }
  if ( *(_DWORD *)(v3 + 1624) <= 0x52u )
  {
    v17 = (_QWORD *)sub_36F41D0(
                      (*(_BYTE *)(*(_QWORD *)(a1 + 256) + 877LL) & 2) != 0,
                      (*(_BYTE *)(*(_QWORD *)(a1 + 256) + 877LL) & 4) != 0);
    return sub_2FF0E80(a1, v17, 0);
  }
  return result;
}
