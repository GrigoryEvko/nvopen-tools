// Function: sub_7E8750
// Address: 0x7e8750
//
__m128i *__fastcall sub_7E8750(__m128i *a1, __int64 a2, int a3)
{
  __m128i *v3; // r13
  __int64 i; // r12
  int v5; // edx
  __int64 v6; // rax
  _QWORD *v7; // r15
  __int64 v8; // rcx
  __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 j; // rax
  __m128i *v12; // rax
  __m128i *v14; // rax
  __int64 v15; // [rsp+0h] [rbp-40h]
  _QWORD *v17; // [rsp+8h] [rbp-38h]

  v3 = a1;
  for ( i = a1->m128i_i64[0]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  sub_7E3EE0(i);
  v5 = a3;
  if ( (*(_BYTE *)(a2 + 96) & 2) != 0 )
    return (__m128i *)sub_7E85E0(a1, (__int64 *)a2, a3);
  v6 = *(_QWORD *)(a2 + 112);
  v7 = *(_QWORD **)(v6 + 8);
  v8 = v7[2];
  v17 = *(_QWORD **)(v6 + 16);
  if ( (*(_BYTE *)(v8 + 96) & 2) != 0 )
  {
    v15 = v7[2];
    v14 = (__m128i *)sub_7E85E0(a1, (__int64 *)v8, v5);
    v7 = (_QWORD *)*v7;
    v3 = v14;
    i = *(_QWORD *)(v15 + 40);
  }
  for ( ; (_QWORD *)*v17 != v7; v3 = v12 )
  {
    v9 = v7[2];
    v10 = v9;
    if ( *(_QWORD **)(*(_QWORD *)(a2 + 112) + 8LL) != v7 )
      v10 = sub_8D5CF0(i, *(_QWORD *)(v9 + 40));
    for ( j = v3->m128i_i64[0]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v12 = (__m128i *)sub_7E2400(v3, *(_QWORD *)(j + 160), *(_QWORD *)(v10 + 104), *(_QWORD *)(v10 + 40));
    i = *(_QWORD *)(v9 + 40);
    v7 = (_QWORD *)*v7;
  }
  return v3;
}
