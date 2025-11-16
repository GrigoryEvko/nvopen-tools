// Function: sub_19C4690
// Address: 0x19c4690
//
__int64 __fastcall sub_19C4690(
        __int64 a1,
        _QWORD *a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __m128 a7,
        double a8,
        double a9,
        double a10,
        double a11,
        double a12,
        double a13,
        __m128 a14)
{
  __int64 v18; // rax
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // rsi
  unsigned __int64 v22; // rax
  __int64 v23; // r13
  double v24; // xmm4_8
  double v25; // xmm5_8
  __int64 v28; // [rsp+10h] [rbp-40h]
  __int64 v29; // [rsp+18h] [rbp-38h]

  v18 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4F9A488, 1u);
  if ( v18 )
  {
    v19 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v18 + 104LL))(v18, &unk_4F9A488);
    if ( v19 )
      sub_1465DB0(*(_QWORD *)(v19 + 160), a2);
  }
  v20 = sub_1AA91E0(*(_QWORD *)(a1 + 320), *(_QWORD *)(a1 + 312), *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 160));
  v21 = *(_QWORD *)(a5 + 48);
  v28 = v20;
  if ( v21 )
    v21 -= 24;
  v29 = sub_1AA8CA0(a5, v21, *(_QWORD *)(a1 + 304), *(_QWORD *)(a1 + 160));
  v22 = sub_157EBA0(*(_QWORD *)(a1 + 320));
  v23 = v22;
  if ( *(_BYTE *)(v22 + 16) == 26 )
  {
    sub_19C0AD0(a1, a3, a4, v29, v28, (_QWORD *)v22, a6);
    sub_14045C0(*(_QWORD *)(a1 + 168), v23, (__int64)a2);
    sub_15F2000(v23);
    sub_1648B90(v23);
  }
  else
  {
    sub_19C0AD0(a1, a3, a4, v29, v28, 0, a6);
    sub_14045C0(*(_QWORD *)(a1 + 168), 0, (__int64)a2);
  }
  *(_BYTE *)(a1 + 289) = 1;
  return sub_19C3980(a1, (__int64)a2, a3, a4, 0, a7, a8, a9, a10, v24, v25, a13, a14);
}
