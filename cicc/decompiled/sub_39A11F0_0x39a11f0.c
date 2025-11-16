// Function: sub_39A11F0
// Address: 0x39a11f0
//
__int64 __fastcall sub_39A11F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  unsigned __int16 v6; // ax

  result = *(unsigned int *)(a1 + 12);
  if ( (_DWORD)result )
  {
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a2 + 256) + 160LL))(*(_QWORD *)(a2 + 256), a3, 0);
    sub_396F340(a2, 4 * *(_DWORD *)(a1 + 12) + 4);
    v6 = sub_3971A70(a2);
    sub_396F320(a2, v6);
    result = sub_396F320(a2, 0);
    if ( a4 )
      return (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a2 + 256) + 176LL))(
               *(_QWORD *)(a2 + 256),
               a4,
               0);
  }
  return result;
}
