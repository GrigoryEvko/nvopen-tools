// Function: sub_9D21E0
// Address: 0x9d21e0
//
void __fastcall __noreturn sub_9D21E0(_BYTE *a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // r12
  void (__fastcall *v5)(__int64, __int64); // rbx
  __int64 v6; // rax
  __int64 v7; // rax

  v2 = sub_C5F790();
  sub_904010(v2, "Expected<T> must be checked before access or destruction.\n");
  if ( (a1[8] & 1) != 0 )
  {
    v3 = sub_C5F790();
    sub_904010(v3, "Unchecked Expected<T> contained error:\n");
    v4 = *(_QWORD *)a1;
    v5 = *(void (__fastcall **)(__int64, __int64))(**(_QWORD **)a1 + 16LL);
    v6 = sub_C5F790();
    v5(v4, v6);
  }
  else
  {
    v7 = sub_C5F790();
    sub_904010(
      v7,
      "Expected<T> value was in success state. (Note: Expected<T> values in success mode must still be checked prior to b"
      "eing destroyed).\n");
  }
  abort();
}
