// Function: sub_A18460
// Address: 0xa18460
//
__int64 __fastcall sub_A18460(__int64 a1, __int64 a2)
{
  __int64 v3; // rdi
  __int64 *v4; // r14
  __int64 *v5; // r12
  __int64 i; // rax
  __int64 v7; // rdi
  unsigned int v8; // ecx
  __int64 *v9; // r12
  __int64 *v10; // r13
  __int64 v11; // rdi
  __int64 v12; // rdi
  __int64 result; // rax
  __int64 v14; // r12

  v3 = *(_QWORD *)(a1 + 160);
  if ( v3 )
  {
    a2 = *(_QWORD *)(a1 + 176) - v3;
    j_j___libc_free_0(v3, a2);
  }
  v4 = *(__int64 **)(a1 + 72);
  v5 = &v4[*(unsigned int *)(a1 + 80)];
  if ( v4 != v5 )
  {
    for ( i = *(_QWORD *)(a1 + 72); ; i = *(_QWORD *)(a1 + 72) )
    {
      v7 = *v4;
      v8 = (unsigned int)(((__int64)v4 - i) >> 3) >> 7;
      a2 = 4096LL << v8;
      if ( v8 >= 0x1E )
        a2 = 0x40000000000LL;
      ++v4;
      sub_C7D6A0(v7, a2, 16);
      if ( v5 == v4 )
        break;
    }
  }
  v9 = *(__int64 **)(a1 + 120);
  v10 = &v9[2 * *(unsigned int *)(a1 + 128)];
  if ( v9 != v10 )
  {
    do
    {
      a2 = v9[1];
      v11 = *v9;
      v9 += 2;
      sub_C7D6A0(v11, a2, 16);
    }
    while ( v10 != v9 );
    v10 = *(__int64 **)(a1 + 120);
  }
  if ( v10 != (__int64 *)(a1 + 136) )
    _libc_free(v10, a2);
  v12 = *(_QWORD *)(a1 + 72);
  if ( v12 != a1 + 88 )
    _libc_free(v12, a2);
  result = sub_C0BF30(a1 + 8);
  v14 = *(_QWORD *)a1;
  if ( *(_QWORD *)a1 )
  {
    sub_A173F0(*(_QWORD *)a1, (_QWORD *)a2);
    return j_j___libc_free_0(v14, 152);
  }
  return result;
}
