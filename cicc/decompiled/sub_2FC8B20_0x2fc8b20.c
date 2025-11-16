// Function: sub_2FC8B20
// Address: 0x2fc8b20
//
__int64 __fastcall sub_2FC8B20(__int64 a1, __int64 a2)
{
  __int64 *v2; // rbx
  __int64 result; // rax
  __int64 *i; // r13
  __int64 v6; // rsi

  v2 = *(__int64 **)(a1 + 112);
  result = 3LL * *(unsigned int *)(a1 + 120);
  for ( i = &v2[3 * *(unsigned int *)(a1 + 120)];
        i != v2;
        result = (*(__int64 (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, *(v2 - 1), 8) )
  {
    v6 = *v2;
    v2 += 3;
    sub_E9A500(a2, v6, 8u, 0);
    (*(void (__fastcall **)(__int64, _QWORD, __int64))(*(_QWORD *)a2 + 536LL))(a2, *(v2 - 2), 8);
  }
  return result;
}
