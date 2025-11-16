// Function: sub_2FC8BA0
// Address: 0x2fc8ba0
//
__int64 __fastcall sub_2FC8BA0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  __int64 i; // r13
  __int64 v5; // rsi
  __int64 result; // rax

  v2 = *(_QWORD *)(a1 + 64);
  for ( i = v2 + 16LL * *(unsigned int *)(a1 + 72);
        v2 != i;
        result = (*(__int64 (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)a2 + 536LL))(a2, v5, 8) )
  {
    v5 = *(_QWORD *)(v2 + 8);
    v2 += 16;
  }
  return result;
}
