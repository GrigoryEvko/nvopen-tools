// Function: sub_14A39A0
// Address: 0x14a39a0
//
__int64 __fastcall sub_14A39A0(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdi
  __int64 result; // rax
  __int64 (__fastcall *v7)(__int64, unsigned int, __int64, __int64, __int64); // r9
  unsigned __int64 v8; // rcx

  v5 = *a1;
  result = a2;
  v7 = *(__int64 (__fastcall **)(__int64, unsigned int, __int64, __int64, __int64))(*(_QWORD *)v5 + 840LL);
  if ( v7 != sub_14A0B30 )
    return ((__int64 (__fastcall *)(__int64, _QWORD))v7)(v5, a2);
  v8 = *(_QWORD *)(a5 + 32);
  if ( a2 > v8 && a2 - v8 != 1 )
  {
    for ( result = (unsigned int)v8;
          (_DWORD)result && (((_DWORD)result - 1) & (unsigned int)result) != 0;
          result = (unsigned int)(result - 1) )
    {
      ;
    }
  }
  return result;
}
