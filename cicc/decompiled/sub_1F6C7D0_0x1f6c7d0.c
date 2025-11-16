// Function: sub_1F6C7D0
// Address: 0x1f6c7d0
//
__int64 __fastcall sub_1F6C7D0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  __int64 v6; // r15
  __int64 (__fastcall *v7)(__int64, __int64, __int64, _QWORD, __int64); // r14
  __int64 v8; // rax

  v6 = *(_QWORD *)(a1 + 48);
  v7 = *(__int64 (__fastcall **)(__int64, __int64, __int64, _QWORD, __int64))(*(_QWORD *)a2 + 264LL);
  v8 = sub_1E0A0C0(*(_QWORD *)(a1 + 32));
  return v7(a2, v8, v6, a3, a4);
}
