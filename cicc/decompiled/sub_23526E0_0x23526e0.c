// Function: sub_23526E0
// Address: 0x23526e0
//
void __fastcall __noreturn sub_23526E0(_BYTE *a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // r12
  void (__fastcall *v6)(__int64, __int64); // rbx
  __int64 v7; // rax
  __int64 v8; // rax

  v3 = sub_C5F790((__int64)a1, a2);
  sub_904010(v3, "Expected<T> must be checked before access or destruction.\n");
  if ( (a1[8] & 1) != 0 )
  {
    v4 = sub_C5F790(v3, (__int64)"Expected<T> must be checked before access or destruction.\n");
    sub_904010(v4, "Unchecked Expected<T> contained error:\n");
    v5 = *(_QWORD *)a1;
    v6 = *(void (__fastcall **)(__int64, __int64))(**(_QWORD **)a1 + 16LL);
    v7 = sub_C5F790(v4, (__int64)"Unchecked Expected<T> contained error:\n");
    v6(v5, v7);
  }
  else
  {
    v8 = sub_C5F790(v3, (__int64)"Expected<T> must be checked before access or destruction.\n");
    sub_904010(
      v8,
      "Expected<T> value was in success state. (Note: Expected<T> values in success mode must still be checked prior to b"
      "eing destroyed).\n");
  }
  abort();
}
