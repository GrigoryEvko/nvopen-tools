// Function: sub_35C9CD0
// Address: 0x35c9cd0
//
void __fastcall sub_35C9CD0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4[5]; // [rsp+8h] [rbp-28h] BYREF

  sub_2E32880(v4, a1);
  if ( !sub_2E322F0(a1, a2) )
    (*(void (__fastcall **)(__int64, __int64, __int64, _QWORD, _QWORD, _QWORD, __int64 *, _QWORD))(*(_QWORD *)a3 + 368LL))(
      a3,
      a1,
      a2,
      0,
      0,
      0,
      v4,
      0);
  if ( v4[0] )
    sub_B91220((__int64)v4, v4[0]);
}
