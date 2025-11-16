// Function: sub_1E61C40
// Address: 0x1e61c40
//
__int64 __fastcall sub_1E61C40(__int64 a1, __int64 a2)
{
  __int64 v3; // rbx
  _QWORD *v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rcx
  int v7; // r8d
  int v8; // r9d
  __int64 v9; // rbx
  __int64 *v10; // r15
  unsigned int v11; // edx
  __int64 v12; // rsi
  __int64 v13; // rax
  __int64 *v14; // r14
  __int64 *v15; // rax
  __int64 v16; // r15
  __int64 *v17; // rbx
  __int64 v18; // rdi
  _QWORD *v19; // rbx
  _QWORD *v20; // r12
  unsigned __int64 v21; // rdi
  __int64 result; // rax
  _QWORD *v23; // rbx
  _QWORD *v24; // r12
  unsigned __int64 v25; // rdi
  char *v27[2]; // [rsp+20h] [rbp-A0h] BYREF
  char v28; // [rsp+30h] [rbp-90h] BYREF
  __int64 v29[2]; // [rsp+50h] [rbp-70h] BYREF
  _QWORD *v30; // [rsp+60h] [rbp-60h]
  __int64 v31; // [rsp+68h] [rbp-58h]
  _QWORD *v32; // [rsp+70h] [rbp-50h]
  __int64 v33; // [rsp+78h] [rbp-48h]
  unsigned int v34; // [rsp+80h] [rbp-40h]
  __int64 v35; // [rsp+88h] [rbp-38h]

  v3 = *(_QWORD *)(a1 + 88);
  sub_1E05DB0(a1 + 48);
  *(_QWORD *)(a1 + 88) = v3;
  *(_DWORD *)(a1 + 8) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_BYTE *)(a1 + 96) = 0;
  *(_DWORD *)(a1 + 100) = 0;
  v4 = (_QWORD *)sub_22077B0(8);
  v31 = 0;
  *v4 = 0;
  v29[0] = (__int64)v4;
  v30 = v4 + 1;
  v29[1] = (__int64)(v4 + 1);
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  sub_1E60EF0(v27, a1, 0);
  sub_1E5E5F0(a1, v27, v5, v6, v7, v8);
  if ( v27[0] != &v28 )
    _libc_free((unsigned __int64)v27[0]);
  sub_1E5FE90((__int64)v29);
  v9 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v9 )
  {
    v10 = *(__int64 **)a1;
    v11 = 1;
    do
    {
      v12 = *v10++;
      v11 = sub_1E60810((__int64)v29, v12, v11, (unsigned __int8 (__fastcall *)(__int64))sub_1E5E510, 0);
    }
    while ( (__int64 *)v9 != v10 );
  }
  sub_1E615D0(v29, a1, 0);
  if ( a2 )
    *(_BYTE *)(a2 + 144) = 1;
  if ( *(_DWORD *)(a1 + 8) )
  {
    v27[0] = 0;
    v13 = sub_22077B0(56);
    v14 = (__int64 *)v13;
    if ( v13 )
    {
      *(_QWORD *)v13 = 0;
      *(_QWORD *)(v13 + 8) = 0;
      *(_DWORD *)(v13 + 16) = 0;
      *(_QWORD *)(v13 + 24) = 0;
      *(_QWORD *)(v13 + 32) = 0;
      *(_QWORD *)(v13 + 40) = 0;
      *(_QWORD *)(v13 + 48) = -1;
    }
    v15 = sub_1E063B0(a1 + 48, (__int64 *)v27);
    v16 = v15[1];
    v17 = v15;
    v15[1] = (__int64)v14;
    if ( v16 )
    {
      v18 = *(_QWORD *)(v16 + 24);
      if ( v18 )
        j_j___libc_free_0(v18, *(_QWORD *)(v16 + 40) - v18);
      j_j___libc_free_0(v16, 56);
      v14 = (__int64 *)v17[1];
    }
    *(_QWORD *)(a1 + 80) = v14;
    sub_1E604F0((__int64)v29, a1, v14);
    if ( v34 )
    {
      v19 = v32;
      v20 = &v32[9 * v34];
      do
      {
        if ( *v19 != -8 && *v19 != -16 )
        {
          v21 = v19[5];
          if ( (_QWORD *)v21 != v19 + 7 )
            _libc_free(v21);
        }
        v19 += 9;
      }
      while ( v20 != v19 );
    }
  }
  else if ( v34 )
  {
    v23 = v32;
    v24 = &v32[9 * v34];
    do
    {
      if ( *v23 != -16 && *v23 != -8 )
      {
        v25 = v23[5];
        if ( (_QWORD *)v25 != v23 + 7 )
          _libc_free(v25);
      }
      v23 += 9;
    }
    while ( v24 != v23 );
  }
  result = j___libc_free_0(v32);
  if ( v29[0] )
    return j_j___libc_free_0(v29[0], (char *)v30 - v29[0]);
  return result;
}
