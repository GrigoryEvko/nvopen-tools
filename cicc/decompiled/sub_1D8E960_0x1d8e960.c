// Function: sub_1D8E960
// Address: 0x1d8e960
//
__int64 __fastcall sub_1D8E960(__int64 a1)
{
  __int64 v1; // rax
  __int64 v2; // rdi

  v1 = sub_160F9A0(*(_QWORD *)(a1 + 8), (__int64)&unk_4FC3606, 1u);
  v2 = v1;
  if ( v1 )
    v2 = (*(__int64 (__fastcall **)(__int64, void *))(*(_QWORD *)v1 + 104LL))(v1, &unk_4FC3606);
  sub_1D8E690(v2);
  return 0;
}
