// Function: sub_2EE8070
// Address: 0x2ee8070
//
__int64 __fastcall sub_2EE8070(__int64 a1)
{
  __int64 v2; // rdi
  __int64 result; // rax
  __int64 v4; // rdi

  *(_QWORD *)a1 = 0;
  *(_DWORD *)(a1 + 344) = 0;
  v2 = *(_QWORD *)(a1 + 400);
  *(_QWORD *)(a1 + 400) = 0;
  if ( v2 )
    result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v2 + 24LL))(v2);
  v4 = *(_QWORD *)(a1 + 408);
  *(_QWORD *)(a1 + 408) = 0;
  if ( v4 )
    return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v4 + 24LL))(v4);
  return result;
}
