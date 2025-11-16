// Function: sub_17E8A90
// Address: 0x17e8a90
//
void __fastcall __noreturn sub_17E8A90(_BYTE *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  __int64 v5; // rdx
  __int64 v6; // rdi
  __int64 v7; // r12
  void (__fastcall *v8)(__int64, __int64); // rbx
  __int64 v9; // rdx
  __int64 v10; // rax
  __int64 v11; // rax

  v4 = sub_16BA580((__int64)a1, a2, a3);
  sub_1263B40(v4, "Expected<T> must be checked before access or destruction.\n");
  if ( (a1[32] & 1) != 0 )
  {
    v6 = sub_16BA580(v4, (__int64)"Expected<T> must be checked before access or destruction.\n", v5);
    sub_1263B40(v6, "Unchecked Expected<T> contained error:\n");
    v7 = *(_QWORD *)a1;
    v8 = *(void (__fastcall **)(__int64, __int64))(**(_QWORD **)a1 + 16LL);
    v10 = sub_16BA580(v6, (__int64)"Unchecked Expected<T> contained error:\n", v9);
    v8(v7, v10);
  }
  else
  {
    v11 = sub_16BA580(v4, (__int64)"Expected<T> must be checked before access or destruction.\n", v5);
    sub_1263B40(
      v11,
      "Expected<T> value was in success state. (Note: Expected<T> values in success mode must still be checked prior to b"
      "eing destroyed).\n");
  }
  abort();
}
