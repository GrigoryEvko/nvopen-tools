// Function: sub_9ACC00
// Address: 0x9acc00
//
bool __fastcall sub_9ACC00(__int64 a1, __int64 a2, __m128i *a3)
{
  unsigned __int8 *v5; // r14
  unsigned __int8 *v6; // r13
  bool result; // al
  unsigned int v8; // r12d
  __int64 v9; // r13
  __int64 v10; // r13
  __int64 v11; // rdi
  bool v12; // cc
  unsigned int v13; // eax
  __int64 v14; // rdi
  __int64 v15; // rdi
  unsigned int v16; // eax
  __int64 v17; // rdi
  bool v18; // [rsp+Fh] [rbp-61h]
  __int64 v19; // [rsp+10h] [rbp-60h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-58h]
  __int64 v21; // [rsp+20h] [rbp-50h] BYREF
  unsigned int v22; // [rsp+28h] [rbp-48h]
  __int64 v23; // [rsp+30h] [rbp-40h]
  int v24; // [rsp+38h] [rbp-38h]

  v5 = (unsigned __int8 *)(*(_QWORD *)a1 & 0xFFFFFFFFFFFFFFF8LL);
  v6 = (unsigned __int8 *)(*(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL);
  if ( sub_99BC70(v5, v6, a3->m128i_i64) || sub_99BC70(v6, v5, a3->m128i_i64) )
    return 1;
  if ( (*(_QWORD *)a2 & 4) == 0 )
  {
    sub_9AC330((__int64)&v21, *(_QWORD *)a2 & 0xFFFFFFFFFFFFFFF8LL, 0, a3);
    if ( *(_DWORD *)(a2 + 16) > 0x40u )
    {
      v15 = *(_QWORD *)(a2 + 8);
      if ( v15 )
        j_j___libc_free_0_0(v15);
    }
    v12 = *(_DWORD *)(a2 + 32) <= 0x40u;
    *(_QWORD *)(a2 + 8) = v21;
    v16 = v22;
    v22 = 0;
    *(_DWORD *)(a2 + 16) = v16;
    if ( v12 || (v17 = *(_QWORD *)(a2 + 24)) == 0 )
    {
      *(_QWORD *)(a2 + 24) = v23;
      *(_DWORD *)(a2 + 32) = v24;
    }
    else
    {
      j_j___libc_free_0_0(v17);
      v12 = v22 <= 0x40;
      *(_QWORD *)(a2 + 24) = v23;
      *(_DWORD *)(a2 + 32) = v24;
      if ( !v12 && v21 )
        j_j___libc_free_0_0(v21);
    }
    *(_QWORD *)a2 |= 4uLL;
  }
  if ( (*(_QWORD *)a1 & 4) == 0 )
  {
    sub_9AC330((__int64)&v21, *(_QWORD *)a1 & 0xFFFFFFFFFFFFFFF8LL, 0, a3);
    if ( *(_DWORD *)(a1 + 16) > 0x40u )
    {
      v11 = *(_QWORD *)(a1 + 8);
      if ( v11 )
        j_j___libc_free_0_0(v11);
    }
    v12 = *(_DWORD *)(a1 + 32) <= 0x40u;
    *(_QWORD *)(a1 + 8) = v21;
    v13 = v22;
    v22 = 0;
    *(_DWORD *)(a1 + 16) = v13;
    if ( v12 || (v14 = *(_QWORD *)(a1 + 24)) == 0 )
    {
      *(_QWORD *)(a1 + 24) = v23;
      *(_DWORD *)(a1 + 32) = v24;
    }
    else
    {
      j_j___libc_free_0_0(v14);
      v12 = v22 <= 0x40;
      *(_QWORD *)(a1 + 24) = v23;
      *(_DWORD *)(a1 + 32) = v24;
      if ( !v12 && v21 )
        j_j___libc_free_0_0(v21);
    }
    *(_QWORD *)a1 |= 4uLL;
  }
  v8 = *(_DWORD *)(a1 + 16);
  v20 = v8;
  if ( v8 <= 0x40 )
  {
    v9 = *(_QWORD *)(a1 + 8);
    goto LABEL_9;
  }
  sub_C43780(&v19, a1 + 8);
  v8 = v20;
  if ( v20 <= 0x40 )
  {
    v9 = v19;
LABEL_9:
    v10 = *(_QWORD *)(a2 + 8) | v9;
    if ( v8 )
      return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v8) == v10;
    return 1;
  }
  sub_C43BD0(&v19, a2 + 8);
  v8 = v20;
  v10 = v19;
  v20 = 0;
  v22 = v8;
  v21 = v19;
  if ( !v8 )
    return 1;
  if ( v8 <= 0x40 )
    return 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v8) == v10;
  result = v8 == (unsigned int)sub_C445E0(&v21);
  if ( v10 )
  {
    v18 = result;
    j_j___libc_free_0_0(v10);
    result = v18;
    if ( v20 > 0x40 )
    {
      if ( v19 )
      {
        j_j___libc_free_0_0(v19);
        return v18;
      }
    }
  }
  return result;
}
