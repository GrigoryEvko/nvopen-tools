// Function: sub_C63C30
// Address: 0xc63c30
//
void __fastcall __noreturn sub_C63C30(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax
  unsigned __int64 v3; // rdi
  void (__fastcall *v4)(unsigned __int64, __int64); // rbx
  __int64 v5; // rsi
  __int64 v6; // rax
  __int64 v7; // rax

  v2 = sub_C5F790((__int64)a1, a2);
  sub_904010(v2, "Program aborted due to an unhandled Error:\n");
  v3 = *a1 & 0xFFFFFFFFFFFFFFFELL;
  if ( v3 )
  {
    v4 = *(void (__fastcall **)(unsigned __int64, __int64))(*(_QWORD *)v3 + 16LL);
    v5 = sub_C5F790(v3, (__int64)"Program aborted due to an unhandled Error:\n");
    v4(v3, v5);
    v6 = sub_C5F790(v3, v5);
    sub_904010(v6, "\n");
  }
  else
  {
    v7 = sub_C5F790(0, (__int64)"Program aborted due to an unhandled Error:\n");
    sub_904010(v7, "Error value was Success. (Note: Success values must still be checked prior to being destroyed).\n");
  }
  abort();
}
