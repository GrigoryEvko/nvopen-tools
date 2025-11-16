// Function: sub_2FE2D40
// Address: 0x2fe2d40
//
unsigned __int64 __fastcall sub_2FE2D40(__int64 *a1, __int64 a2, unsigned int a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // r14
  unsigned __int64 result; // rax
  unsigned int v12[20]; // [rsp+20h] [rbp-50h] BYREF

  v7 = *(_QWORD *)(sub_2E88D60(a2) + 32);
  (*(void (__fastcall **)(__int64 *, __int64, _QWORD, unsigned int *))(*a1 + 688))(a1, a2, a3, v12);
  result = sub_2EBEE90(v7, *(_DWORD *)(*(_QWORD *)(a2 + 32) + 40LL * v12[0] + 8));
  if ( *(_QWORD *)(result + 24) == *(_QWORD *)(a2 + 24) )
    return sub_2FE1E80(a1, a2, result, a3, a4, a5, v12, 5, a6);
  return result;
}
