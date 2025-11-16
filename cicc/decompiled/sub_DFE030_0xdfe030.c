// Function: sub_DFE030
// Address: 0xdfe030
//
bool __fastcall sub_DFE030(__int64 *a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rdi
  bool (__fastcall *v5)(__int64, __int64, __int64); // rax
  __int64 v6; // rbx
  __int64 v7; // r8
  bool result; // al
  __int64 v9; // rbx

  v4 = *a1;
  v5 = *(bool (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v4 + 1456LL);
  if ( v5 != sub_DF6A50 )
    return v5(v4, a2, a3);
  v6 = sub_B2D7E0(a3, "target-cpu", 0xAu);
  v7 = sub_B2D7E0(a2, "target-cpu", 0xAu);
  result = 0;
  if ( v6 == v7 )
  {
    v9 = sub_B2D7E0(a3, "target-features", 0xFu);
    return v9 == sub_B2D7E0(a2, "target-features", 0xFu);
  }
  return result;
}
