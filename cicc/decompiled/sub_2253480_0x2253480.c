// Function: sub_2253480
// Address: 0x2253480
//
void __fastcall __noreturn sub_2253480(__int64 a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rax
  _DWORD *v5; // rax
  _QWORD *v6; // rbp
  int v7; // edx
  int v8; // ecx
  int v9; // r8d
  int v10; // r9d

  v4 = sub_22529C0();
  ++*(_DWORD *)(v4 + 8);
  v5 = (_DWORD *)sub_2253430(a1, a2, a3);
  *v5 = 1;
  v6 = v5 + 24;
  sub_39F8140((_DWORD)v5 + 96, a2, v7, v8, v9, v10);
  sub_2252810(v6);
  sub_2207530();
}
