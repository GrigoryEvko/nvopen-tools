// Function: sub_3910950
// Address: 0x3910950
//
__int64 __fastcall sub_3910950(__int64 a1, __int64 a2, int a3, int a4, int a5, __int64 a6, __int64 a7)
{
  __int64 v9; // rcx
  __int64 v12; // rax
  __int64 result; // rax
  __int64 v14; // rbx
  __int64 v15; // [rsp+8h] [rbp-38h]

  v9 = 0;
  v12 = *(unsigned int *)(a2 + 120);
  if ( (_DWORD)v12 )
    v9 = *(_QWORD *)(*(_QWORD *)(a2 + 112) + 32 * v12 - 32);
  v15 = v9;
  result = sub_22077B0(0x68u);
  v14 = result;
  if ( result )
  {
    sub_38CF760(result, 11, 0, v15);
    *(_DWORD *)(v14 + 48) = a3;
    *(_DWORD *)(v14 + 52) = a4;
    *(_QWORD *)(v14 + 72) = a7;
    *(_QWORD *)(v14 + 80) = v14 + 96;
    *(_DWORD *)(v14 + 56) = a5;
    *(_QWORD *)(v14 + 64) = a6;
    *(_QWORD *)(v14 + 88) = 0x800000000LL;
    return 0x800000000LL;
  }
  return result;
}
