// Function: sub_32468F0
// Address: 0x32468f0
//
__int64 __fastcall sub_32468F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 result; // rax
  unsigned __int16 v6; // ax

  result = *(unsigned int *)(a1 + 56);
  if ( (_DWORD)result )
  {
    (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a2 + 224) + 176LL))(*(_QWORD *)(a2 + 224), a3, 0);
    sub_31DF6B0(a2);
    sub_31F0F40(a2);
    v6 = sub_31DF670(a2);
    sub_31DC9F0(a2, v6);
    result = sub_31DC9F0(a2, 0);
    if ( a4 )
      return (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(a2 + 224) + 208LL))(
               *(_QWORD *)(a2 + 224),
               a4,
               0);
  }
  return result;
}
