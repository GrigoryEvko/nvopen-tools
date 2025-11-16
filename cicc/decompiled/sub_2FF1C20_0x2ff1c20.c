// Function: sub_2FF1C20
// Address: 0x2ff1c20
//
__int64 __fastcall sub_2FF1C20(__int64 a1)
{
  _QWORD *v1; // rsi
  __int64 (*v2)(); // rax
  _QWORD *v3; // rax

  v1 = (_QWORD *)(*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 424LL))(a1, 1);
  sub_2FF0E80(a1, v1, 0);
  v2 = *(__int64 (**)())(*(_QWORD *)a1 + 336LL);
  if ( v2 != sub_2FEDAE0 )
    ((void (__fastcall *)(__int64))v2)(a1);
  sub_2FF12A0(a1, &unk_502A65C, 0);
  v3 = (_QWORD *)sub_3595C00();
  sub_2FF0E80(a1, v3, 0);
  return 1;
}
