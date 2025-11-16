// Function: ctor_654
// Address: 0x59b020
//
int ctor_654()
{
  qword_5039148 = (__int64)"vliw-td";
  qword_5039150 = 7;
  qword_5039158 = (__int64)"VLIW scheduler";
  qword_5039140 = unk_5039AB0;
  qword_5039160 = 14;
  qword_5039168 = (__int64)sub_3363950;
  unk_5039AB0 = &qword_5039140;
  if ( unk_5039AC0 )
    (*(void (__fastcall **)(_QWORD, const char *, __int64, __int64 (__fastcall *)(_QWORD, _QWORD), const char *, __int64))(*unk_5039AC0 + 24LL))(
      unk_5039AC0,
      "vliw-td",
      7,
      sub_3363950,
      "VLIW scheduler",
      14);
  return __cxa_atexit(sub_334CAC0, &qword_5039140, &qword_4A427C0);
}
