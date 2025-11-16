// Function: sub_35B4EE0
// Address: 0x35b4ee0
//
__int64 __fastcall sub_35B4EE0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  __int64 v4; // rdx
  __int64 v5; // rsi
  int v6; // [rsp-1Ch] [rbp-1Ch] BYREF

  result = *(unsigned int *)(a2 + 112);
  if ( !*(_DWORD *)(*(_QWORD *)(*(_QWORD *)(a1 + 24) + 32LL) + 4LL * (*(_DWORD *)(a2 + 112) & 0x7FFFFFFF)) )
  {
    if ( !*(_QWORD *)(a1 + 384) )
      return (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, a2);
    v4 = *(_QWORD *)(a1 + 16);
    v5 = *(_QWORD *)(a1 + 8);
    v6 = result;
    result = (*(__int64 (__fastcall **)(__int64, __int64, __int64, int *))(a1 + 392))(a1 + 368, v5, v4, &v6);
    if ( (_BYTE)result )
      return (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 40LL))(a1, a2);
  }
  return result;
}
