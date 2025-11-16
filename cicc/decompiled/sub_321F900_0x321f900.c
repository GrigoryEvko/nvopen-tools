// Function: sub_321F900
// Address: 0x321f900
//
__int64 __fastcall sub_321F900(__int64 a1)
{
  __int64 v2; // r12
  int v3; // r9d
  __int64 v5; // [rsp-10h] [rbp-20h]

  v2 = *(_QWORD *)(sub_31DA6B0(*(_QWORD *)(a1 + 8)) + 224);
  (*(void (__fastcall **)(_QWORD, __int64, _QWORD))(**(_QWORD **)(*(_QWORD *)(a1 + 8) + 224LL) + 176LL))(
    *(_QWORD *)(*(_QWORD *)(a1 + 8) + 224LL),
    v2,
    0);
  sub_3725050(*(_QWORD *)(a1 + 8), a1 + 6016, (int)"types", 5, *(_QWORD *)(v2 + 16), v3, &unk_44D5F10, 3);
  return v5;
}
