// Function: sub_1E70870
// Address: 0x1e70870
//
__int64 __fastcall sub_1E70870(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // r13
  __int64 *v8; // rbx
  __int64 v9; // rsi
  __int64 i; // rbx
  __int64 v11; // rsi
  __int64 result; // rax

  v6 = &a2[a3];
  *(_QWORD *)(a1 + 2264) = 0;
  *(_QWORD *)(a1 + 2256) = 0;
  if ( a2 != v6 )
  {
    v8 = a2;
    do
    {
      v9 = *v8++;
      (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 2120) + 120LL))(*(_QWORD *)(a1 + 2120), v9);
    }
    while ( v6 != v8 );
  }
  for ( i = a4 + 8 * a5; a4 != i; i -= 8 )
  {
    v11 = *(_QWORD *)(i - 8);
    (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 2120) + 128LL))(*(_QWORD *)(a1 + 2120), v11);
  }
  sub_1E704A0(a1, a1 + 72);
  sub_1E70570(a1, a1 + 344);
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 2120) + 88LL))(*(_QWORD *)(a1 + 2120));
  *(_QWORD *)(a1 + 2240) = sub_1E6BEE0(*(_QWORD *)(a1 + 928), *(_QWORD *)(a1 + 936));
  result = *(_QWORD *)(a1 + 936);
  *(_QWORD *)(a1 + 2248) = result;
  return result;
}
