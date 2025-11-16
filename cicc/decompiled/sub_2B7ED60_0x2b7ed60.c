// Function: sub_2B7ED60
// Address: 0x2b7ed60
//
__int64 __fastcall sub_2B7ED60(__int64 a1, __int64 *a2, _QWORD *a3)
{
  __int64 v4; // [rsp-10h] [rbp-30h]
  _QWORD v5[3]; // [rsp+8h] [rbp-18h] BYREF

  sub_2B4E520(*(unsigned int ***)a1, *(_QWORD *)(a1 + 8), a3, 0);
  *a2 = (__int64)sub_2B7E230(
                   *(_QWORD *)(*(_QWORD *)(a1 + 16) + 120LL),
                   **(_QWORD **)(a1 + 8),
                   *(unsigned int *)(*(_QWORD *)(a1 + 8) + 8LL),
                   *a2,
                   **(_QWORD **)(a1 + 16),
                   *(unsigned int *)(*(_QWORD *)(a1 + 8) + 8LL),
                   (__int64 (__fastcall *)(__int64, __int64, _QWORD *, __int64 *, _QWORD))sub_2B7B2F0,
                   (__int64)v5);
  return v4;
}
