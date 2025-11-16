// Function: sub_2FF3D20
// Address: 0x2ff3d20
//
__int64 __fastcall sub_2FF3D20(__int64 a1)
{
  _DWORD *v1; // rax
  _QWORD *v2; // rsi
  __int64 v4; // r14
  void (__fastcall *v5)(__int64, __m128i *, __int64); // rbx
  __m128i *v6; // rax
  _QWORD *v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rsi
  __int64 (__fastcall *v10)(__int64); // rax
  _QWORD *v11; // rax
  _QWORD *v12; // rax
  __m128i v13; // [rsp+0h] [rbp-40h] BYREF
  void (__fastcall *v14)(__m128i *, __m128i *, __int64); // [rsp+10h] [rbp-30h]

  v1 = sub_C94E20((__int64)qword_4F86310);
  if ( v1 )
  {
    if ( *v1 != 2 )
    {
LABEL_3:
      sub_2FF3AB0(a1, (__int64)&unk_503FCFC, 0, 0);
      sub_2FF3AB0(a1, (__int64)&unk_50201DC, 0, 0);
      sub_2FF3AB0(a1, (__int64)&unk_501F390, 0, 0);
      sub_2FF3AB0(a1, (__int64)&unk_5026411, 0, 0);
      sub_2FF3AB0(a1, (__int64)&unk_5040114, 0, 0);
      sub_2FF3AB0(a1, (__int64)&unk_503A0EC, 0, 0);
      sub_2FF3AB0(a1, (__int64)&unk_5021D24, 0, 0);
      sub_2FF3AB0(a1, (__int64)&unk_5022FAC, 0, 0);
      sub_2FF3AB0(a1, (__int64)&unk_503B12C, 0, 0);
      sub_2FF3AB0(a1, (__int64)&unk_50226EC, 0, 0);
      sub_2FF3AB0(a1, (__int64)&unk_503FF48, 0, 0);
      sub_2FF3AB0(a1, (__int64)&unk_503FF3C, 0, 0);
      v2 = (_QWORD *)sub_25D70E0();
      sub_2FF0E80(a1, v2, 0);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 176LL))(a1);
      return sub_2FF2920((_BYTE *)a1);
    }
  }
  else if ( LODWORD(qword_4F86310[2]) != 2 )
  {
    goto LABEL_3;
  }
  if ( sub_23CF310(*(_QWORD *)(a1 + 256)) )
  {
    v11 = (_QWORD *)sub_2E2F870();
    sub_2FF0E80(a1, v11, 0);
  }
  v4 = *(_QWORD *)(a1 + 176);
  v5 = *(void (__fastcall **)(__int64, __m128i *, __int64))(*(_QWORD *)v4 + 16LL);
  sub_23CF4B0(&v13, *(_QWORD *)(a1 + 256));
  v6 = sub_DFEEB0(&v13);
  v5(v4, v6, 1);
  if ( v14 )
    v14(&v13, &v13, 3);
  v7 = (_QWORD *)sub_2F3C630();
  sub_2FF0E80(a1, v7, 0);
  v8 = (_QWORD *)sub_2DBA350();
  sub_2FF0E80(a1, v8, 0);
  v9 = (_QWORD *)sub_2DBBC40();
  sub_2FF0E80(a1, v9, 0);
  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
  v10 = *(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 168LL);
  if ( v10 == sub_2FF1980 )
  {
    if ( (unsigned int)sub_2FF0570(a1) )
    {
      if ( !byte_5029108 )
      {
        v12 = (_QWORD *)sub_2D5CDB0();
        sub_2FF0E80(a1, v12, 0);
      }
    }
  }
  else
  {
    v10(a1);
  }
  sub_2FF1E40(a1);
  return 1;
}
