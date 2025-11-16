// Function: sub_35F4AD0
// Address: 0x35f4ad0
//
__int64 __fastcall sub_35F4AD0(__int64 a1, __int64 a2, unsigned int a3, __int64 a4)
{
  void *v4; // rdx

  if ( (*(_BYTE *)(*(_QWORD *)(a2 + 16) + 16LL * a3 + 8) & 0xF) != 0 )
    sub_C64ED0("Invalid opcode for nvvm_fence_opcode intrinsic", 1u);
  v4 = *(void **)(a4 + 32);
  if ( *(_QWORD *)(a4 + 24) - (_QWORD)v4 <= 0xDu )
    return sub_CB6200(a4, ".mbarrier_init", 0xEu);
  qmemcpy(v4, ".mbarrier_init", 14);
  *(_QWORD *)(a4 + 32) += 14LL;
  return 29801;
}
