// Function: sub_22534D0
// Address: 0x22534d0
//
void __fastcall __noreturn sub_22534D0(__int64 a1, int a2)
{
  unsigned __int64 v2; // rax
  int v3; // ecx
  int v4; // r8d
  int v5; // r9d
  __int64 v6; // rdi
  unsigned __int64 v7; // rdx

  v2 = sub_22529C0();
  v6 = *(_QWORD *)v2;
  ++*(_DWORD *)(v2 + 8);
  if ( v6 )
  {
    v7 = *(_QWORD *)(v6 + 80) - 0x474E5543432B2B00LL;
    if ( v7 <= 1 )
      *(_DWORD *)(v6 + 40) = -*(_DWORD *)(v6 + 40);
    else
      *(_QWORD *)v2 = 0;
    sub_39F8800(v6 + 80, a2, v7, v3, v4, v5);
    sub_2252810((_QWORD *)(v6 + 80));
  }
  sub_2207530();
}
