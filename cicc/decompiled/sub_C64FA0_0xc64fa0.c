// Function: sub_C64FA0
// Address: 0xc64fa0
//
void __fastcall __noreturn sub_C64FA0(const char *a1, char *a2, unsigned int a3)
{
  const char *v3; // r12
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rsi
  __int64 v8; // rdi
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  __int64 v12; // rax

  v3 = a2;
  if ( a1 )
  {
    v5 = sub_C5F790((__int64)a1, (__int64)a2);
    a2 = "\n";
    a1 = (const char *)sub_904010(v5, a1);
    sub_904010((__int64)a1, "\n");
  }
  v6 = sub_C5F790((__int64)a1, (__int64)a2);
  v7 = (__int64)"UNREACHABLE executed";
  v8 = v6;
  sub_904010(v6, "UNREACHABLE executed");
  if ( v3 )
  {
    v9 = sub_C5F790(v8, (__int64)"UNREACHABLE executed");
    v10 = sub_904010(v9, " at ");
    v11 = sub_904010(v10, v3);
    v7 = a3;
    v8 = sub_904010(v11, ":");
    sub_CB59D0(v8, a3);
  }
  v12 = sub_C5F790(v8, v7);
  sub_904010(v12, "!\n");
  abort();
}
