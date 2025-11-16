// Function: sub_C6ED20
// Address: 0xc6ed20
//
__int64 __fastcall sub_C6ED20(__int64 a1, __int64 a2)
{
  unsigned int v2; // eax
  unsigned int v3; // r14d
  __int64 v4; // r13
  unsigned int v5; // edx
  unsigned __int64 v6; // rax
  __int64 v7; // rcx
  __int64 v8; // rax
  unsigned int v9; // eax
  const void *v10; // rdx
  unsigned int v11; // eax
  bool v12; // cc
  __int64 v14; // r13
  unsigned int v15; // eax
  const void *v16; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v17; // [rsp+8h] [rbp-58h]
  unsigned __int64 v18; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-48h]
  const void *v20; // [rsp+20h] [rbp-40h] BYREF
  unsigned int v21; // [rsp+28h] [rbp-38h]
  unsigned __int64 v22; // [rsp+30h] [rbp-30h] BYREF
  unsigned int v23; // [rsp+38h] [rbp-28h]

  v2 = *(_DWORD *)(a2 + 8);
  v3 = v2 - 1;
  v17 = *(_DWORD *)(a2 + 24);
  if ( v17 > 0x40 )
  {
    sub_C43780((__int64)&v16, (const void **)(a2 + 16));
    v2 = *(_DWORD *)(a2 + 8);
  }
  else
  {
    v16 = *(const void **)(a2 + 16);
  }
  v19 = v2;
  v4 = 1LL << v3;
  if ( v2 <= 0x40 )
  {
    v18 = *(_QWORD *)a2;
    goto LABEL_5;
  }
  sub_C43780((__int64)&v18, (const void **)a2);
  if ( *(_DWORD *)(a2 + 8) <= 0x40u )
  {
LABEL_5:
    v5 = v17;
    v6 = (unsigned __int64)v16;
    if ( (v4 & *(_QWORD *)a2) != 0 )
      goto LABEL_6;
    goto LABEL_10;
  }
  v5 = v17;
  v6 = (unsigned __int64)v16;
  if ( (v4 & *(_QWORD *)(*(_QWORD *)a2 + 8LL * (v3 >> 6))) != 0 )
  {
LABEL_6:
    if ( v5 > 0x40 )
      *(_QWORD *)(v6 + 8LL * (v3 >> 6)) |= v4;
    else
      v16 = (const void *)(v4 | v6);
    goto LABEL_12;
  }
LABEL_10:
  v7 = ~v4;
  if ( v5 <= 0x40 )
    v16 = (const void *)(v7 & v6);
  else
    *(_QWORD *)(v6 + 8LL * (v3 >> 6)) &= v7;
LABEL_12:
  v8 = *(_QWORD *)(a2 + 16);
  if ( *(_DWORD *)(a2 + 24) > 0x40u )
    v8 = *(_QWORD *)(v8 + 8LL * (v3 >> 6));
  if ( (v4 & v8) != 0 )
  {
    if ( v19 <= 0x40 )
    {
      v18 |= v4;
LABEL_17:
      v23 = v19;
LABEL_18:
      v22 = v18;
      goto LABEL_19;
    }
    *(_QWORD *)(v18 + 8LL * (v3 >> 6)) |= v4;
    v15 = v19;
  }
  else
  {
    v14 = ~v4;
    if ( v19 <= 0x40 )
    {
      v18 &= v14;
      goto LABEL_17;
    }
    *(_QWORD *)(v18 + 8LL * (v3 >> 6)) &= v14;
    v15 = v19;
  }
  v23 = v15;
  if ( v15 <= 0x40 )
    goto LABEL_18;
  sub_C43780((__int64)&v22, (const void **)&v18);
LABEL_19:
  v9 = v17;
  v21 = v17;
  if ( v17 > 0x40 )
  {
    sub_C43780((__int64)&v20, &v16);
    v9 = v21;
    v10 = v20;
  }
  else
  {
    v10 = v16;
  }
  *(_DWORD *)(a1 + 8) = v9;
  v11 = v23;
  v12 = v19 <= 0x40;
  *(_QWORD *)a1 = v10;
  *(_DWORD *)(a1 + 24) = v11;
  *(_QWORD *)(a1 + 16) = v22;
  if ( !v12 && v18 )
    j_j___libc_free_0_0(v18);
  if ( v17 > 0x40 && v16 )
    j_j___libc_free_0_0(v16);
  return a1;
}
