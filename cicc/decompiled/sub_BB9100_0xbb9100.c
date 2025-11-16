// Function: sub_BB9100
// Address: 0xbb9100
//
__int64 __fastcall sub_BB9100(__int64 a1)
{
  _QWORD *v2; // r13
  _QWORD *v3; // rbx
  _QWORD *v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rsi
  _QWORD *v7; // rbx
  _QWORD *v8; // rdi
  __int64 v9; // rdi
  __int64 result; // rax
  __int64 v11; // rdi

  v2 = *(_QWORD **)(a1 + 8);
  *(_QWORD *)a1 = &unk_49DAD88;
  if ( v2 )
  {
    if ( *v2 )
      j_j___libc_free_0(*v2, v2[2] - *v2);
    j_j___libc_free_0(v2, 32);
  }
  v3 = *(_QWORD **)(a1 + 128);
  while ( v3 )
  {
    v4 = v3;
    v3 = (_QWORD *)*v3;
    j_j___libc_free_0(v4, 16);
  }
  memset(*(void **)(a1 + 112), 0, 8LL * *(_QWORD *)(a1 + 120));
  v5 = *(_QWORD *)(a1 + 112);
  v6 = *(_QWORD *)(a1 + 120);
  *(_QWORD *)(a1 + 136) = 0;
  *(_QWORD *)(a1 + 128) = 0;
  if ( v5 != a1 + 160 )
    j_j___libc_free_0(v5, 8 * v6);
  v7 = *(_QWORD **)(a1 + 72);
  while ( v7 )
  {
    v8 = v7;
    v7 = (_QWORD *)*v7;
    j_j___libc_free_0(v8, 16);
  }
  memset(*(void **)(a1 + 56), 0, 8LL * *(_QWORD *)(a1 + 64));
  v9 = *(_QWORD *)(a1 + 56);
  result = a1 + 104;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  if ( v9 != a1 + 104 )
    result = j_j___libc_free_0(v9, 8LL * *(_QWORD *)(a1 + 64));
  v11 = *(_QWORD *)(a1 + 32);
  if ( v11 )
    return j_j___libc_free_0(v11, *(_QWORD *)(a1 + 48) - v11);
  return result;
}
