// Function: sub_C70170
// Address: 0xc70170
//
__int64 __fastcall sub_C70170(__int64 a1, __int64 a2)
{
  unsigned int v2; // r14d
  __int64 v3; // r13
  unsigned int v4; // edx
  unsigned __int64 v5; // rax
  __int64 v6; // rcx
  unsigned __int64 v7; // rax
  unsigned int v8; // eax
  const void *v9; // rdx
  unsigned int v10; // eax
  bool v11; // cc
  __int64 v13; // r13
  unsigned int v14; // eax
  const void *v15; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v16; // [rsp+8h] [rbp-58h]
  unsigned __int64 v17; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v18; // [rsp+18h] [rbp-48h]
  const void *v19; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v20; // [rsp+28h] [rbp-38h]
  unsigned __int64 v21; // [rsp+30h] [rbp-30h] BYREF
  unsigned int v22; // [rsp+38h] [rbp-28h]

  v16 = *(_DWORD *)(a2 + 8);
  v2 = v16 - 1;
  if ( v16 > 0x40 )
    sub_C43780((__int64)&v15, (const void **)a2);
  else
    v15 = *(const void **)a2;
  v3 = 1LL << v2;
  v18 = *(_DWORD *)(a2 + 24);
  if ( v18 <= 0x40 )
  {
    v17 = *(_QWORD *)(a2 + 16);
    goto LABEL_5;
  }
  sub_C43780((__int64)&v17, (const void **)(a2 + 16));
  if ( *(_DWORD *)(a2 + 24) <= 0x40u )
  {
LABEL_5:
    v4 = v16;
    v5 = (unsigned __int64)v15;
    if ( (v3 & *(_QWORD *)(a2 + 16)) != 0 )
      goto LABEL_6;
    goto LABEL_10;
  }
  v4 = v16;
  v5 = (unsigned __int64)v15;
  if ( (v3 & *(_QWORD *)(*(_QWORD *)(a2 + 16) + 8LL * (v2 >> 6))) != 0 )
  {
LABEL_6:
    if ( v4 > 0x40 )
      *(_QWORD *)(v5 + 8LL * (v2 >> 6)) |= v3;
    else
      v15 = (const void *)(v3 | v5);
    goto LABEL_12;
  }
LABEL_10:
  v6 = ~v3;
  if ( v4 <= 0x40 )
    v15 = (const void *)(v6 & v5);
  else
    *(_QWORD *)(v5 + 8LL * (v2 >> 6)) &= v6;
LABEL_12:
  v7 = *(_QWORD *)a2;
  if ( *(_DWORD *)(a2 + 8) > 0x40u )
    v7 = *(_QWORD *)(v7 + 8LL * (v2 >> 6));
  if ( (v3 & v7) != 0 )
  {
    if ( v18 <= 0x40 )
    {
      v17 |= v3;
LABEL_17:
      v22 = v18;
LABEL_18:
      v21 = v17;
      goto LABEL_19;
    }
    *(_QWORD *)(v17 + 8LL * (v2 >> 6)) |= v3;
    v14 = v18;
  }
  else
  {
    v13 = ~v3;
    if ( v18 <= 0x40 )
    {
      v17 &= v13;
      goto LABEL_17;
    }
    *(_QWORD *)(v17 + 8LL * (v2 >> 6)) &= v13;
    v14 = v18;
  }
  v22 = v14;
  if ( v14 <= 0x40 )
    goto LABEL_18;
  sub_C43780((__int64)&v21, (const void **)&v17);
LABEL_19:
  v8 = v16;
  v20 = v16;
  if ( v16 > 0x40 )
  {
    sub_C43780((__int64)&v19, &v15);
    v9 = v19;
    v8 = v20;
  }
  else
  {
    v9 = v15;
  }
  *(_DWORD *)(a1 + 8) = v8;
  v10 = v22;
  v11 = v18 <= 0x40;
  *(_QWORD *)a1 = v9;
  *(_DWORD *)(a1 + 24) = v10;
  *(_QWORD *)(a1 + 16) = v21;
  if ( !v11 && v17 )
    j_j___libc_free_0_0(v17);
  if ( v16 > 0x40 && v15 )
    j_j___libc_free_0_0(v15);
  return a1;
}
