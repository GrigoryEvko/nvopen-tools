// Function: sub_14A3BF0
// Address: 0x14a3bf0
//
__int64 __fastcall sub_14A3BF0(__int64 a1)
{
  bool v1; // zf
  void (__fastcall *v2)(__int64, __int64, __int64); // rax

  v1 = *(_BYTE *)(a1 + 200) == 0;
  *(_QWORD *)a1 = &unk_49ECA68;
  if ( !v1 )
    sub_14A3B20((_QWORD *)(a1 + 192));
  v2 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 176);
  if ( v2 )
    v2(a1 + 160, a1 + 160, 3);
  sub_16367B0(a1);
  return j_j___libc_free_0(a1, 208);
}
