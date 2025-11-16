// Function: sub_2EC65D0
// Address: 0x2ec65d0
//
__int64 __fastcall sub_2EC65D0(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 *v6; // r13
  __int64 *v8; // rbx
  __int64 v9; // rsi
  __int64 i; // rbx
  __int64 v11; // rsi
  __int64 result; // rax

  v6 = &a2[a3];
  *(_QWORD *)(a1 + 3528) = 0;
  *(_QWORD *)(a1 + 3520) = 0;
  if ( a2 != v6 )
  {
    v8 = a2;
    do
    {
      v9 = *v8++;
      (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 3472) + 128LL))(*(_QWORD *)(a1 + 3472), v9);
    }
    while ( v6 != v8 );
  }
  for ( i = a4 + 8 * a5; a4 != i; i -= 8 )
  {
    v11 = *(_QWORD *)(i - 8);
    (*(void (__fastcall **)(_QWORD, __int64))(**(_QWORD **)(a1 + 3472) + 136LL))(*(_QWORD *)(a1 + 3472), v11);
  }
  sub_2EC61D0(a1, a1 + 72);
  sub_2EC62A0(a1, a1 + 328);
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 3472) + 96LL))(*(_QWORD *)(a1 + 3472));
  *(_QWORD *)(a1 + 3504) = sub_2EC2050(*(_QWORD *)(a1 + 912), *(_QWORD *)(a1 + 920));
  result = *(_QWORD *)(a1 + 920);
  *(_QWORD *)(a1 + 3512) = result;
  return result;
}
