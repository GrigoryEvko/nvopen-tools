// Function: sub_15C7E40
// Address: 0x15c7e40
//
__int64 __fastcall sub_15C7E40(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v3; // rax

  (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a2 + 128LL))(a2, *(_QWORD *)(a1 + 24));
  result = *(unsigned int *)(a1 + 16);
  if ( (_DWORD)result )
  {
    v3 = (*(__int64 (__fastcall **)(__int64, char *))(*(_QWORD *)a2 + 48LL))(a2, " at line ");
    return (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v3 + 104LL))(v3, *(unsigned int *)(a1 + 16));
  }
  return result;
}
