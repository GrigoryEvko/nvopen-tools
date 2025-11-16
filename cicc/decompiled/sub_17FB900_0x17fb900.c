// Function: sub_17FB900
// Address: 0x17fb900
//
__int64 __fastcall sub_17FB900(__int64 a1, __int64 *a2)
{
  __int64 v2; // rax
  int v3; // r9d
  __int64 v4; // rax

  v2 = sub_1632FA0((__int64)a2);
  *(_QWORD *)(a1 + 160) = sub_15A9620(v2, *a2, 0);
  v4 = sub_1B281E0(
         (_DWORD)a2,
         (unsigned int)"tsan.module_ctor",
         16,
         (unsigned int)"__tsan_init",
         11,
         v3,
         0,
         0,
         0,
         0,
         0,
         0);
  *(_QWORD *)(a1 + 984) = v4;
  sub_1B28000(a2, v4, 0, 0);
  return 1;
}
