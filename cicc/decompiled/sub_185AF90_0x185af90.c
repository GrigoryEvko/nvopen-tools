// Function: sub_185AF90
// Address: 0x185af90
//
__int64 __fastcall sub_185AF90(unsigned __int64 *a1, unsigned __int64 a2)
{
  __int64 v2; // rax

  v2 = sub_161ACC0(*(_QWORD *)(*a1 + 8), *a1, (__int64)&unk_4F9E06C, a2);
  return (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 104LL))(v2, &unk_4F9E06C) + 160;
}
