// Function: sub_35E4D30
// Address: 0x35e4d30
//
__int64 __fastcall sub_35E4D30(__int64 a1, __int64 a2, unsigned int a3)
{
  __int64 result; // rax
  int v5; // ebx
  unsigned int v6; // edx

  result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 72LL))(a1);
  v5 = result;
  if ( dword_5040368 != (_DWORD)result )
  {
    v6 = (*(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)a1 + 80LL))(a1, a3, (unsigned int)result);
    result = v5 + v6 + 1;
    if ( dword_5040368 == v6 )
      return v6;
  }
  return result;
}
