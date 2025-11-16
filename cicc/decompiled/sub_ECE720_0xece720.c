// Function: sub_ECE720
// Address: 0xece720
//
__int64 __fastcall sub_ECE720(__int64 a1)
{
  __int64 v1; // r12
  __int64 v2; // rax

  v1 = *(_QWORD *)(*(_QWORD *)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8))
                             + 168)
                 + 32LL);
  v2 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v2 + 176LL))(v2, v1, 0);
  return 0;
}
