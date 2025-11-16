// Function: ctor_686
// Address: 0x5a61e0
//
int ctor_686()
{
  qword_503FDE8 = (__int64)"basic";
  qword_503FDF0 = 5;
  qword_503FDF8 = (__int64)"basic register allocator";
  qword_503FDE0 = unk_5023860;
  qword_503FE00 = 24;
  qword_503FE08 = (__int64)sub_35B6440;
  unk_5023860 = &qword_503FDE0;
  if ( qword_5023870 )
    (*(void (__fastcall **)(_QWORD *, char *, __int64, __int64 (*)(void), const char *, __int64))(*qword_5023870 + 24LL))(
      qword_5023870,
      "basic",
      5,
      sub_35B6440,
      "basic register allocator",
      24);
  return __cxa_atexit(sub_2F41140, &qword_503FDE0, &qword_4A427C0);
}
