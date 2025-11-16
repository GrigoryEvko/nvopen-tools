// Function: sub_2EC1090
// Address: 0x2ec1090
//
__int64 __fastcall sub_2EC1090(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned int a5)
{
  __int64 result; // rax
  char v9; // [rsp+6h] [rbp-2Ah]

  sub_2F90C80();
  (*(void (__fastcall **)(_QWORD, __int64, __int64, _QWORD))(**(_QWORD **)(a1 + 3472) + 24LL))(
    *(_QWORD *)(a1 + 3472),
    a3,
    a4,
    a5);
  v9 = (*(unsigned int (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 3472) + 32LL))(*(_QWORD *)(a1 + 3472)) >> 16;
  result = 0;
  if ( !v9 )
    result = (unsigned int)(((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 3472) + 32LL))(*(_QWORD *)(a1 + 3472))
                           & 0xFF000000) == 0)
           + 1;
  *(_DWORD *)(a1 + 2912) = result;
  return result;
}
