// Function: sub_E775C0
// Address: 0xe775c0
//
__int64 __fastcall sub_E775C0(__int64 a1, _QWORD *a2, int a3, __int64 a4)
{
  __int64 *v5; // rbx
  __int64 v6; // rdx
  __int64 v7; // r13
  __int64 *v8; // r12
  __int64 v9; // rsi
  int **v10; // rdx

  sub_E77580((__int64 *)a1, a2, (unsigned __int16)a3 | (BYTE2(a3) << 16), a4);
  v5 = *(__int64 **)(a1 + 552);
  v7 = v6;
  v8 = &v5[4 * *(unsigned int *)(a1 + 560)];
  while ( v8 != v5 )
  {
    v9 = *v5;
    v10 = (int **)(v5 + 1);
    v5 += 4;
    sub_E761F0(a2, v9, v10);
  }
  return (*(__int64 (__fastcall **)(_QWORD *, __int64, _QWORD))(*a2 + 208LL))(a2, v7, 0);
}
