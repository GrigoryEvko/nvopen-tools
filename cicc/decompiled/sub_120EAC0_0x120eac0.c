// Function: sub_120EAC0
// Address: 0x120eac0
//
__int64 __fastcall sub_120EAC0(__int64 a1)
{
  __int64 i; // r12
  __int64 v3; // r13
  __int64 v4; // rsi
  __int64 j; // r12
  __int64 v6; // r13
  __int64 v7; // rsi
  __int64 result; // rax
  __int64 k; // r12
  __int64 v10; // rdi
  __int64 m; // rbx
  __int64 v12; // r12
  __int64 v13; // rdi

  for ( i = *(_QWORD *)(a1 + 40); a1 + 24 != i; i = sub_220EEE0(i) )
  {
    v3 = *(_QWORD *)(i + 64);
    if ( *(_BYTE *)v3 != 23 )
    {
      v4 = sub_ACADE0(*(__int64 ***)(v3 + 8));
      sub_BD84D0(v3, v4);
      sub_BD72D0(*(_QWORD *)(i + 64), v4);
    }
  }
  for ( j = *(_QWORD *)(a1 + 88); a1 + 72 != j; j = sub_220EEE0(j) )
  {
    v6 = *(_QWORD *)(j + 40);
    if ( *(_BYTE *)v6 != 23 )
    {
      v7 = sub_ACADE0(*(__int64 ***)(v6 + 8));
      sub_BD84D0(v6, v7);
      sub_BD72D0(*(_QWORD *)(j + 40), v7);
    }
  }
  result = sub_C7D6A0(*(_QWORD *)(a1 + 120), 16LL * *(unsigned int *)(a1 + 136), 8);
  for ( k = *(_QWORD *)(a1 + 80); k; result = j_j___libc_free_0(v10, 56) )
  {
    sub_1206180(*(_QWORD *)(k + 24));
    v10 = k;
    k = *(_QWORD *)(k + 16);
  }
  for ( m = *(_QWORD *)(a1 + 32); m; result = j_j___libc_free_0(v12, 80) )
  {
    v12 = m;
    sub_1207330(*(_QWORD **)(m + 24));
    v13 = *(_QWORD *)(m + 32);
    m = *(_QWORD *)(m + 16);
    if ( v13 != v12 + 48 )
      j_j___libc_free_0(v13, *(_QWORD *)(v12 + 48) + 1LL);
  }
  return result;
}
