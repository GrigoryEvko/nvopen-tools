// Function: sub_15C7B30
// Address: 0x15c7b30
//
__int64 __fastcall sub_15C7B30(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax

  v2 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a2 + 48LL))(a2, a1[3]);
  (*(void (__fastcall **)(__int64, const char *))(*(_QWORD *)v2 + 48LL))(v2, " limit");
  if ( a1[5] )
  {
    v3 = (*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)a2 + 48LL))(a2, " of ");
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v3 + 64LL))(v3, a1[5]);
  }
  v4 = (*(__int64 (__fastcall **)(__int64, const char *))(*(_QWORD *)a2 + 48LL))(a2, " exceeded (");
  v5 = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v4 + 64LL))(v4, a1[4]);
  v6 = (*(__int64 (__fastcall **)(__int64, const char *))(*(_QWORD *)v5 + 48LL))(v5, ") in ");
  return (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v6 + 136LL))(v6, a1[2]);
}
