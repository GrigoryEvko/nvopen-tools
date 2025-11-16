// Function: sub_261CAC0
// Address: 0x261cac0
//
__int64 __fastcall sub_261CAC0(__int64 a1, _QWORD *a2)
{
  char v3; // [rsp+7h] [rbp-29h] BYREF
  _QWORD v4[5]; // [rsp+8h] [rbp-28h] BYREF

  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "GUID",
         0,
         0,
         &v3,
         v4) )
  {
    sub_261BC10(a1, a2);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v4[0]);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Offset",
         0,
         0,
         &v3,
         v4) )
  {
    sub_261BC10(a1, a2 + 1);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v4[0]);
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
}
