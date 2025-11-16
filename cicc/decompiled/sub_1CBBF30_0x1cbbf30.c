// Function: sub_1CBBF30
// Address: 0x1cbbf30
//
__int64 __fastcall sub_1CBBF30(__int64 a1)
{
  _QWORD *v2; // rbx
  _QWORD *v3; // rdi
  __int64 v4; // rdi
  _QWORD *v5; // rbx
  _QWORD *v6; // rdi
  __int64 v7; // rdi
  __int64 v8; // rsi
  __int64 result; // rax
  __int64 i; // rbx
  __int64 v11; // rdi
  __int64 j; // rbx
  __int64 v13; // rdi

  v2 = *(_QWORD **)(a1 + 176);
  while ( v2 )
  {
    v3 = v2;
    v2 = (_QWORD *)*v2;
    j_j___libc_free_0(v3, 16);
  }
  memset(*(void **)(a1 + 160), 0, 8LL * *(_QWORD *)(a1 + 168));
  v4 = *(_QWORD *)(a1 + 160);
  *(_QWORD *)(a1 + 184) = 0;
  *(_QWORD *)(a1 + 176) = 0;
  if ( v4 != a1 + 208 )
    j_j___libc_free_0(v4, 8LL * *(_QWORD *)(a1 + 168));
  v5 = *(_QWORD **)(a1 + 120);
  while ( v5 )
  {
    v6 = v5;
    v5 = (_QWORD *)*v5;
    j_j___libc_free_0(v6, 16);
  }
  memset(*(void **)(a1 + 104), 0, 8LL * *(_QWORD *)(a1 + 112));
  v7 = *(_QWORD *)(a1 + 104);
  v8 = *(_QWORD *)(a1 + 112);
  result = a1 + 152;
  *(_QWORD *)(a1 + 128) = 0;
  *(_QWORD *)(a1 + 120) = 0;
  if ( v7 != a1 + 152 )
    result = j_j___libc_free_0(v7, 8 * v8);
  for ( i = *(_QWORD *)(a1 + 72); i; result = j_j___libc_free_0(v11, 40) )
  {
    sub_1CBB400(*(_QWORD *)(i + 24));
    v11 = i;
    i = *(_QWORD *)(i + 16);
  }
  for ( j = *(_QWORD *)(a1 + 24); j; result = j_j___libc_free_0(v13, 40) )
  {
    sub_1CBB400(*(_QWORD *)(j + 24));
    v13 = j;
    j = *(_QWORD *)(j + 16);
  }
  return result;
}
