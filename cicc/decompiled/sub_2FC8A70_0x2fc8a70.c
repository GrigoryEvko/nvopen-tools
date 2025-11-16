// Function: sub_2FC8A70
// Address: 0x2fc8a70
//
__int64 __fastcall sub_2FC8A70(__int64 a1, __int64 a2)
{
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, (int)qword_5026048, 1);
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, 0, 1);
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, 0, 2);
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, *(unsigned int *)(a1 + 120), 4);
  (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, *(unsigned int *)(a1 + 72), 4);
  return (*(__int64 (__fastcall **)(__int64, unsigned __int64, __int64))(*(_QWORD *)a2 + 536LL))(
           a2,
           0xAAAAAAAAAAAAAAABLL * ((__int64)(*(_QWORD *)(a1 + 16) - *(_QWORD *)(a1 + 8)) >> 6),
           4);
}
