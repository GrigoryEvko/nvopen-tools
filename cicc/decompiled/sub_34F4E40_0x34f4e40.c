// Function: sub_34F4E40
// Address: 0x34f4e40
//
__m128i *__fastcall sub_34F4E40(__int64 a1)
{
  __int64 v1; // r13
  unsigned int v2; // ebx
  unsigned int v3; // eax
  __int64 v4; // r8
  __int64 v5; // r9

  v1 = *(unsigned int *)(**(_QWORD **)(a1 + 24) + 8LL);
  v2 = sub_2E8E690(*(_QWORD *)(a1 + 8));
  v3 = sub_2E8E690(*(_QWORD *)(a1 + 16));
  return sub_2E79810(
           *(_QWORD *)(*(_QWORD *)a1 + 8LL),
           (v1 << 32) | v3,
           ((unsigned __int64)unk_445066C << 32) | v2,
           0,
           v4,
           v5);
}
