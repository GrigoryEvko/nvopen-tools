// Function: sub_E5B460
// Address: 0xe5b460
//
__int64 __fastcall sub_E5B460(
        __int64 a1,
        void (__fastcall ***a2)(_QWORD, _QWORD, __int64, __int64, _QWORD),
        unsigned int a3)
{
  void (__fastcall ***v5)(_QWORD, _QWORD, __int64, __int64, _QWORD); // rsi
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rcx

  v5 = 0;
  v7 = *(unsigned int *)(a1 + 128);
  if ( (_DWORD)v7 )
  {
    v7 = *(_QWORD *)(a1 + 120) + 32 * v7 - 32;
    v5 = *(void (__fastcall ****)(_QWORD, _QWORD, __int64, __int64, _QWORD))v7;
    LODWORD(v7) = *(_DWORD *)(v7 + 8);
  }
  if ( !*(_BYTE *)(a1 + 744) || a2 != v5 || a3 != (_DWORD)v7 )
  {
    v8 = *(_QWORD *)(a1 + 16);
    v9 = *(_QWORD *)(a1 + 304);
    *(_BYTE *)(a1 + 744) = 1;
    if ( v8 )
      (*(void (__fastcall **)(__int64, _QWORD, _QWORD, _QWORD, __int64))(*(_QWORD *)v8 + 48LL))(v8, v5, a2, a3, v9);
    else
      (**a2)(a2, *(_QWORD *)(a1 + 312), *(_QWORD *)(a1 + 8) + 24LL, v9, a3);
  }
  return sub_E980F0(a1, a2, a3);
}
