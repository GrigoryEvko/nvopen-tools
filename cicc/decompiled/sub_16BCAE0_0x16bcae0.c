// Function: sub_16BCAE0
// Address: 0x16bcae0
//
void __fastcall __noreturn sub_16BCAE0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  unsigned __int64 v5; // rdi
  void (__fastcall *v6)(unsigned __int64, __int64); // rbx
  __int64 v7; // rax
  __int64 v8; // rax

  v3 = sub_16BA580((__int64)a1, a2, a3);
  sub_1263B40(v3, "Program aborted due to an unhandled Error:\n");
  v5 = *a1 & 0xFFFFFFFFFFFFFFFELL;
  if ( v5 )
  {
    v6 = *(void (__fastcall **)(unsigned __int64, __int64))(*(_QWORD *)v5 + 16LL);
    v7 = sub_16BA580(v5, (__int64)"Program aborted due to an unhandled Error:\n", v4);
    v6(v5, v7);
  }
  else
  {
    v8 = sub_16BA580(0, (__int64)"Program aborted due to an unhandled Error:\n", v4);
    sub_1263B40(v8, "Error value was Success. (Note: Success values must still be checked prior to being destroyed).\n");
  }
  abort();
}
