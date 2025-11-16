// Function: sub_1D920D0
// Address: 0x1d920d0
//
__int64 __fastcall sub_1D920D0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r13d
  __int64 v4; // rdi
  __int64 (*v5)(); // rax
  __int64 v7; // rdx
  __int64 v8[5]; // [rsp+8h] [rbp-28h] BYREF

  v2 = 0;
  v4 = *(_QWORD *)(a1 + 544);
  v8[0] = 0;
  v5 = *(__int64 (**)())(*(_QWORD *)v4 + 624LL);
  if ( v5 == sub_1D918B0 )
    return v2;
  if ( !((unsigned __int8 (__fastcall *)(__int64, __int64))v5)(v4, a2 + 40) )
  {
    v2 = 1;
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD))(**(_QWORD **)(a1 + 544) + 280LL))(
      *(_QWORD *)(a1 + 544),
      *(_QWORD *)(a2 + 16),
      0);
    (*(void (__fastcall **)(_QWORD, _QWORD, _QWORD, _QWORD, _QWORD, _QWORD, __int64 *, _QWORD))(**(_QWORD **)(a1 + 544)
                                                                                              + 288LL))(
      *(_QWORD *)(a1 + 544),
      *(_QWORD *)(a2 + 16),
      *(_QWORD *)(a2 + 32),
      *(_QWORD *)(a2 + 24),
      *(_QWORD *)(a2 + 40),
      *(unsigned int *)(a2 + 48),
      v8,
      0);
    v7 = *(_QWORD *)(a2 + 32);
    *(_QWORD *)(a2 + 32) = *(_QWORD *)(a2 + 24);
    *(_QWORD *)(a2 + 24) = v7;
  }
  if ( !v8[0] )
    return v2;
  sub_161E7C0((__int64)v8, v8[0]);
  return v2;
}
