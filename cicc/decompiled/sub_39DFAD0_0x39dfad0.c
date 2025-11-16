// Function: sub_39DFAD0
// Address: 0x39dfad0
//
__int64 __fastcall sub_39DFAD0(
        __int64 a1,
        __int64 (__fastcall ***a2)(_QWORD, _QWORD, __int64, __int64, __int64),
        __int64 a3)
{
  __int64 v4; // r12
  __int64 v5; // r9
  __int64 v6; // rsi
  __int64 v7; // rax

  v4 = *(_QWORD *)(a1 + 16);
  v5 = *(_QWORD *)(a1 + 272);
  if ( !v4 )
    return (**a2)(a2, *(_QWORD *)(a1 + 280), *(_QWORD *)(*(_QWORD *)(a1 + 8) + 32LL) + 696LL, v5, a3);
  v6 = 0;
  v7 = *(unsigned int *)(a1 + 120);
  if ( (_DWORD)v7 )
    v6 = *(_QWORD *)(*(_QWORD *)(a1 + 112) + 32 * v7 - 32);
  return (*(__int64 (__fastcall **)(_QWORD, __int64, _QWORD, __int64, __int64))(*(_QWORD *)v4 + 48LL))(
           *(_QWORD *)(a1 + 16),
           v6,
           a2,
           a3,
           v5);
}
