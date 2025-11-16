// Function: sub_3242070
// Address: 0x3242070
//
__int64 __fastcall sub_3242070(__int64 a1, unsigned __int64 a2)
{
  __int64 (__fastcall *v2)(__int64, __int64, _QWORD); // rax

  v2 = *(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 8LL);
  if ( a2 <= 0x1F )
    return v2(a1, (unsigned __int8)(a2 + 48), 0);
  if ( a2 == -1 )
  {
    v2(a1, 48, 0);
    return (*(__int64 (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)a1 + 8LL))(a1, 32, 0);
  }
  else
  {
    v2(a1, 16, 0);
    return (*(__int64 (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)a1 + 24LL))(a1, a2);
  }
}
