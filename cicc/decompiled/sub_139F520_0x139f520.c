// Function: sub_139F520
// Address: 0x139f520
//
__int64 __fastcall sub_139F520(__int64 a1, unsigned int a2, int a3, __int64 a4)
{
  __int64 v7; // rax
  int v8; // r14d
  __int64 v9; // rax
  __int64 v10; // rdi
  unsigned int v11; // edx
  __int64 result; // rax
  __int64 v13; // r12
  __int64 v14; // rdi
  unsigned int v15; // eax
  __int64 v16; // [rsp-10h] [rbp-70h]
  __int64 v17; // [rsp+8h] [rbp-58h]
  __int64 v18; // [rsp+8h] [rbp-58h]
  __int64 v19; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v20; // [rsp+18h] [rbp-48h]
  __int64 v21; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v22; // [rsp+28h] [rbp-38h]

  v7 = sub_15F2050(**(_QWORD **)a1);
  v20 = a2;
  v8 = sub_1632FA0(v7);
  if ( a2 > 0x40 )
  {
    sub_16A4EF0(&v19, 0, 0);
    v22 = a2;
    sub_16A4EF0(&v21, 0, 0);
  }
  else
  {
    v19 = 0;
    v22 = a2;
    v21 = 0;
  }
  v9 = *(_QWORD *)(a1 + 8);
  if ( *(_DWORD *)(v9 + 8) > 0x40u && *(_QWORD *)v9 )
  {
    v17 = *(_QWORD *)(a1 + 8);
    j_j___libc_free_0_0(*(_QWORD *)v9);
    v9 = v17;
  }
  *(_QWORD *)v9 = v19;
  *(_DWORD *)(v9 + 8) = v20;
  v20 = 0;
  if ( *(_DWORD *)(v9 + 24) > 0x40u && (v10 = *(_QWORD *)(v9 + 16)) != 0 )
  {
    v18 = v9;
    j_j___libc_free_0_0(v10);
    v11 = v20;
    *(_QWORD *)(v18 + 16) = v21;
    *(_DWORD *)(v18 + 24) = v22;
    if ( v11 > 0x40 && v19 )
      j_j___libc_free_0_0(v19);
  }
  else
  {
    *(_QWORD *)(v9 + 16) = v21;
    *(_DWORD *)(v9 + 24) = v22;
  }
  result = sub_14BB090(
             a3,
             *(_QWORD *)(a1 + 8),
             v8,
             0,
             *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL),
             **(_QWORD **)(a1 + 24),
             *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL),
             0);
  if ( a4 )
  {
    v20 = a2;
    if ( a2 <= 0x40 )
    {
      v19 = 0;
      v22 = a2;
      v21 = 0;
    }
    else
    {
      sub_16A4EF0(&v19, 0, 0);
      v22 = a2;
      sub_16A4EF0(&v21, 0, 0);
    }
    v13 = *(_QWORD *)(a1 + 32);
    if ( *(_DWORD *)(v13 + 8) > 0x40u && *(_QWORD *)v13 )
      j_j___libc_free_0_0(*(_QWORD *)v13);
    *(_QWORD *)v13 = v19;
    *(_DWORD *)(v13 + 8) = v20;
    v20 = 0;
    if ( *(_DWORD *)(v13 + 24) > 0x40u && (v14 = *(_QWORD *)(v13 + 16)) != 0 )
    {
      j_j___libc_free_0_0(v14);
      v15 = v20;
      *(_QWORD *)(v13 + 16) = v21;
      *(_DWORD *)(v13 + 24) = v22;
      if ( v15 > 0x40 )
      {
        if ( v19 )
          j_j___libc_free_0_0(v19);
      }
    }
    else
    {
      *(_QWORD *)(v13 + 16) = v21;
      *(_DWORD *)(v13 + 24) = v22;
    }
    sub_14BB090(
      a4,
      *(_QWORD *)(a1 + 32),
      v8,
      0,
      *(_QWORD *)(*(_QWORD *)(a1 + 16) + 8LL),
      **(_QWORD **)(a1 + 24),
      *(_QWORD *)(*(_QWORD *)(a1 + 16) + 16LL),
      0);
    return v16;
  }
  return result;
}
