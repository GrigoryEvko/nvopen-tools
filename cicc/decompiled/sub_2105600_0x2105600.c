// Function: sub_2105600
// Address: 0x2105600
//
_BOOL8 __fastcall sub_2105600(__int64 a1, __int64 *a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  int v7; // r8d
  int v8; // r9d

  return !(unsigned __int8)sub_1636800(a1, a2)
      && (v3 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4FCBA30, 1u)) != 0
      && (v4 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v3 + 104LL))(v3, &unk_4FCBA30)) != 0
      && (unsigned __int8)sub_17006E0(*(_QWORD *)(v4 + 208))
      && sub_2104B30(a1, (__int64)a2, v5, v6, v7, v8);
}
