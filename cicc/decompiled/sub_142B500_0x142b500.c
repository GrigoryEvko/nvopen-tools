// Function: sub_142B500
// Address: 0x142b500
//
__int64 __fastcall sub_142B500(_QWORD *a1, __int64 a2)
{
  __int64 v2; // rax

  v2 = sub_161ACC0(*(_QWORD *)(*a1 + 8LL), *a1, &unk_4F97E48, a2);
  return (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v2 + 104LL))(v2, &unk_4F97E48) + 160;
}
