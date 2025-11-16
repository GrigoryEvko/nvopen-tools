// Function: sub_2BDC7C0
// Address: 0x2bdc7c0
//
__int64 __fastcall sub_2BDC7C0(_QWORD **a1, char *a2)
{
  unsigned int v2; // r12d
  __int64 v3; // rax
  __int64 v4; // rsi
  __int64 v5; // rax
  char v6; // r13
  __int64 v7; // rax
  unsigned int v8; // eax
  char v9; // r8
  int v10; // edx

  v2 = *a2;
  v3 = sub_222F790(*a1, (__int64)a2);
  v4 = v2;
  LOBYTE(v2) = (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)v3 + 32LL))(v3, v2);
  v5 = sub_222F790(*a1, v4);
  v6 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v5 + 32LL))(v5, 10);
  v7 = sub_222F790(*a1, 10);
  v8 = (*(__int64 (__fastcall **)(__int64, __int64))(*(_QWORD *)v7 + 32LL))(v7, 13);
  v9 = v8;
  LOBYTE(v8) = v6 != (char)v2;
  LOBYTE(v10) = v9 != (char)v2;
  return v10 & v8;
}
