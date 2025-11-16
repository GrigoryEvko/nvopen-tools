// Function: sub_E7D800
// Address: 0xe7d800
//
__int64 __fastcall sub_E7D800(_QWORD *a1, char a2, __int64 a3)
{
  __int64 v6; // r14
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 (__fastcall *v9)(_QWORD *, __int64, __int64, _QWORD); // r15
  __int64 (*v10)(); // rax
  __int64 result; // rax
  __int64 v12; // rdi
  __int64 v13; // rsi
  __int64 (__fastcall *v14)(_QWORD *, __int64, _QWORD); // rbx
  __int64 (*v15)(); // rax
  __int64 v16; // rax

  v6 = a1[1];
  (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD))(*a1 + 176LL))(a1, *(_QWORD *)(*(_QWORD *)(v6 + 168) + 24LL), 0);
  v7 = 2;
  v8 = *(_QWORD *)(v6 + 168);
  v9 = *(__int64 (__fastcall **)(_QWORD *, __int64, __int64, _QWORD))(*a1 + 616LL);
  v10 = *(__int64 (**)())(*(_QWORD *)v8 + 16LL);
  if ( v10 != sub_E7D6F0 )
  {
    LODWORD(v16) = ((__int64 (__fastcall *)(__int64, __int64))v10)(v8, 2);
    if ( (_DWORD)v16 )
    {
      _BitScanReverse64((unsigned __int64 *)&v16, (unsigned int)v16);
      v7 = 63 - ((unsigned int)v16 ^ 0x3F);
    }
    else
    {
      v7 = 0xFFFFFFFFLL;
    }
  }
  result = v9(a1, v7, a3, 0);
  if ( a2 )
  {
    v12 = *(_QWORD *)(v6 + 152);
    v13 = 0;
    v14 = *(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD))(*a1 + 176LL);
    v15 = *(__int64 (**)())(*(_QWORD *)v12 + 16LL);
    if ( v15 != sub_E7D6D0 )
      v13 = ((__int64 (__fastcall *)(__int64, __int64))v15)(v12, v6);
    return v14(a1, v13, 0);
  }
  return result;
}
