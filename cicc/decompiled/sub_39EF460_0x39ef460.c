// Function: sub_39EF460
// Address: 0x39ef460
//
__int64 __fastcall sub_39EF460(_QWORD *a1, char a2)
{
  __int64 v3; // r13
  __int64 result; // rax
  __int64 v5; // rdi
  __int64 v6; // rsi
  __int64 (__fastcall *v7)(_QWORD *, __int64, _QWORD); // rbx
  __int64 (*v8)(); // rax

  v3 = a1[1];
  (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*a1 + 160LL))(a1, *(_QWORD *)(*(_QWORD *)(v3 + 32) + 24LL), 0);
  result = (*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD))(*a1 + 520LL))(a1, 4, 0);
  if ( a2 )
  {
    v5 = *(_QWORD *)(v3 + 16);
    v6 = 0;
    v7 = *(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD))(*a1 + 160LL);
    v8 = *(__int64 (**)())(*(_QWORD *)v5 + 16LL);
    if ( v8 != sub_21BC3B0 )
      v6 = ((__int64 (__fastcall *)(__int64, __int64))v8)(v5, v3);
    return v7(a1, v6, 0);
  }
  return result;
}
