// Function: sub_88A1F0
// Address: 0x88a1f0
//
__int64 sub_88A1F0()
{
  __int64 v0; // rax
  __int64 result; // rax

  v0 = sub_88A060();
  result = sub_888EB0("__builtin_va_list", v0);
  qword_4D04980 = (_QWORD *)result;
  *(_BYTE *)(result + 143) |= 0x10u;
  return result;
}
