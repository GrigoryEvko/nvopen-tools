// Function: sub_15C7C60
// Address: 0x15c7c60
//
__int64 __fastcall sub_15C7C60(__int64 a1, __int64 a2)
{
  __int64 v3; // rax

  if ( *(_QWORD *)(a1 + 24) )
  {
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a2 + 40LL))(a2, *(_QWORD *)(a1 + 16));
    if ( *(_DWORD *)(a1 + 32) )
    {
      v3 = (*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)a2 + 48LL))(a2, ":");
      (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v3 + 104LL))(v3, *(unsigned int *)(a1 + 32));
    }
    (*(void (__fastcall **)(__int64, char *))(*(_QWORD *)a2 + 48LL))(a2, ": ");
  }
  return (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a2 + 128LL))(a2, *(_QWORD *)(a1 + 40));
}
