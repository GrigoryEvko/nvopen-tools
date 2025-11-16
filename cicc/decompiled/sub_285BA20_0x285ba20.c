// Function: sub_285BA20
// Address: 0x285ba20
//
bool __fastcall sub_285BA20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v7; // rcx
  __int64 v9; // rdx
  bool result; // al
  __int64 *v11; // rax

  if ( *(_DWORD *)(a2 + 32) != 2 )
    return 0;
  v7 = *(_QWORD *)(a2 + 40);
  v9 = *(unsigned __int8 *)(v7 + 8);
  if ( (unsigned int)(v9 - 17) <= 1 )
    v9 = *(unsigned __int8 *)(**(_QWORD **)(v7 + 16) + 8LL);
  result = (_BYTE)v9 == 12
        && *(_WORD *)(a3 + 24) == 8
        && !*(_WORD *)(sub_D33D80((_QWORD *)a3, a5, v9, v7, a5) + 24)
        && ((sub_D95540(**(_QWORD **)(a3 + 32)), (unsigned __int8)sub_DFE0F0(a1))
         || (sub_D95540(**(_QWORD **)(a3 + 32)), (unsigned __int8)sub_DFE120(a1)))
        && (v11 = *(__int64 **)(a3 + 32), *(_WORD *)(*v11 + 24))
        && sub_DADE90(a5, *v11, a4);
  return result;
}
