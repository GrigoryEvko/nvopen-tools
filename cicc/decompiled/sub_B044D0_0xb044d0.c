// Function: sub_B044D0
// Address: 0xb044d0
//
__int64 __fastcall sub_B044D0(__int64 a1, int a2, __int64 a3)
{
  __int64 result; // rax
  unsigned int v4; // r12d
  int v6; // esi
  __int64 *v7; // rsi
  __int64 v8; // rdi
  int v9; // eax
  __int64 v10; // rbx
  int v11; // r14d
  unsigned int v12; // r12d
  unsigned int v13; // edx
  __int64 *v14; // r8
  __int64 v15; // rcx
  int v16; // r9d
  __int64 *v17; // r10
  int v18; // eax
  __int64 v19; // [rsp+8h] [rbp-58h] BYREF
  __int64 *v20; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v21; // [rsp+18h] [rbp-48h]
  __int64 v22; // [rsp+20h] [rbp-40h] BYREF
  bool v23; // [rsp+28h] [rbp-38h]

  v19 = a1;
  if ( a2 )
  {
    result = a1;
    if ( a2 == 1 )
    {
      sub_B95A20(a1);
      return v19;
    }
    return result;
  }
  v4 = *(_DWORD *)(a3 + 24);
  if ( !v4 )
  {
    ++*(_QWORD *)a3;
    v20 = 0;
    goto LABEL_7;
  }
  v10 = *(_QWORD *)(a3 + 8);
  v21 = *(_DWORD *)(a1 + 24);
  if ( v21 > 0x40 )
    sub_C43780(&v20, a1 + 16);
  else
    v20 = *(__int64 **)(a1 + 16);
  v22 = sub_AF5140(a1, 0);
  v23 = *(_DWORD *)(a1 + 4) != 0;
  v11 = sub_AFB7E0((__int64)&v20, &v22);
  if ( v21 > 0x40 && v20 )
    j_j___libc_free_0_0(v20);
  v12 = v4 - 1;
  v8 = v19;
  v13 = v12 & v11;
  v14 = (__int64 *)(v10 + 8LL * (v12 & v11));
  result = v19;
  v15 = *v14;
  if ( *v14 != v19 )
  {
    v16 = 1;
    v7 = 0;
    while ( v15 != -4096 )
    {
      if ( v15 != -8192 || v7 )
        v14 = v7;
      v13 = v12 & (v16 + v13);
      v17 = (__int64 *)(v10 + 8LL * v13);
      v15 = *v17;
      if ( *v17 == v19 )
        return result;
      ++v16;
      v7 = v14;
      v14 = (__int64 *)(v10 + 8LL * v13);
    }
    v18 = *(_DWORD *)(a3 + 16);
    v4 = *(_DWORD *)(a3 + 24);
    if ( !v7 )
      v7 = v14;
    ++*(_QWORD *)a3;
    v9 = v18 + 1;
    v20 = v7;
    if ( 4 * v9 < 3 * v4 )
    {
      if ( v4 - (v9 + *(_DWORD *)(a3 + 20)) > v4 >> 3 )
        goto LABEL_9;
      v6 = v4;
LABEL_8:
      sub_B04210(a3, v6);
      sub_AFCD70(a3, &v19, &v20);
      v7 = v20;
      v8 = v19;
      v9 = *(_DWORD *)(a3 + 16) + 1;
LABEL_9:
      *(_DWORD *)(a3 + 16) = v9;
      if ( *v7 != -4096 )
        --*(_DWORD *)(a3 + 20);
      *v7 = v8;
      return v19;
    }
LABEL_7:
    v6 = 2 * v4;
    goto LABEL_8;
  }
  return result;
}
