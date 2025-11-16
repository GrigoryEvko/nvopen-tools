// Function: sub_B19380
// Address: 0xb19380
//
__int64 *__fastcall sub_B19380(__int64 a1, __int64 a2)
{
  __int64 v2; // r9
  __int64 *result; // rax
  __int64 v4; // rdx
  __int64 v5; // r10
  _QWORD *v6; // r12
  __int64 *v7; // rax
  __int64 v8; // r11
  __int64 *v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r10
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // r12
  __int64 v15; // rdi
  __int64 v16; // [rsp+8h] [rbp-18h] BYREF

  v2 = 0;
  if ( a2 )
    v2 = 8LL * (unsigned int)(*(_DWORD *)(a2 + 44) + 1);
  result = (__int64 *)(v2 + *(_QWORD *)(a1 + 24));
  v4 = *result;
  *(_BYTE *)(a1 + 112) = 0;
  v5 = *(_QWORD *)(v4 + 8);
  v16 = v4;
  if ( v5 )
  {
    v6 = *(_QWORD **)(v5 + 24);
    v7 = sub_B186E0(v6, (__int64)&v6[*(unsigned int *)(v5 + 32)], &v16);
    v9 = (_QWORD *)((char *)v6 + v8 - 8);
    v10 = *v7;
    a2 = *v9;
    *v7 = *v9;
    *v9 = v10;
    --*(_DWORD *)(v11 + 32);
    result = (__int64 *)(v13 + *(_QWORD *)(v12 + 24));
  }
  v14 = *result;
  *result = 0;
  if ( v14 )
  {
    v15 = *(_QWORD *)(v14 + 24);
    if ( v15 != v14 + 40 )
      _libc_free(v15, a2);
    return (__int64 *)j_j___libc_free_0(v14, 80);
  }
  return result;
}
