// Function: sub_14A3A00
// Address: 0x14a3a00
//
__int64 __fastcall sub_14A3A00(__int64 *a1, unsigned int a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v5; // rdi
  __int64 result; // rax
  __int64 (__fastcall *v7)(__int64, unsigned int, __int64, __int64, __int64); // r9
  unsigned __int64 v8; // rdx

  v5 = *a1;
  result = a2;
  v7 = *(__int64 (__fastcall **)(__int64, unsigned int, __int64, __int64, __int64))(*(_QWORD *)v5 + 848LL);
  if ( v7 != sub_14A0B60 )
    return ((__int64 (__fastcall *)(__int64, _QWORD))v7)(v5, a2);
  v8 = *(_QWORD *)(a5 + 32);
  if ( a2 > v8 )
  {
    do
    {
      result = (unsigned int)v8;
      if ( !(_DWORD)v8 )
        break;
      LODWORD(v8) = v8 - 1;
    }
    while ( ((unsigned int)v8 & (unsigned int)result) != 0 );
  }
  return result;
}
