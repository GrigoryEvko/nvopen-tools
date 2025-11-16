// Function: sub_3215490
// Address: 0x3215490
//
__int64 __fastcall sub_3215490(_QWORD *a1, __int64 a2, __int16 a3)
{
  __int64 (__fastcall *v4)(__int64, _QWORD, _QWORD); // r14
  unsigned __int64 v5; // rax
  unsigned int v6; // eax
  int v8; // [rsp+Ah] [rbp-26h] BYREF
  __int16 v9; // [rsp+Eh] [rbp-22h]

  v4 = *(__int64 (__fastcall **)(__int64, _QWORD, _QWORD))(*(_QWORD *)a2 + 456LL);
  v5 = sub_31DF6E0(a2);
  v8 = v5;
  v9 = WORD2(v5);
  v6 = sub_3215440((__int64)a1, (__int64)&v8, a3);
  return v4(a2, *a1, v6);
}
