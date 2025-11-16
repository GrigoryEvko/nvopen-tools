// Function: sub_398A760
// Address: 0x398a760
//
__int64 __fastcall sub_398A760(__int64 a1)
{
  __int64 v2; // r12
  int v3; // r9d
  __int64 v5; // [rsp-10h] [rbp-20h]

  v2 = *(_QWORD *)(sub_396DD80(*(_QWORD *)(a1 + 8)) + 216);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 256LL) + 160LL))(
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 256LL),
    v2,
    0);
  sub_39BD6B0(*(_QWORD *)(a1 + 8), a1 + 6352, (int)"types", 5, *(_QWORD *)(v2 + 8), v3, &unk_4533F70, 3);
  return v5;
}
