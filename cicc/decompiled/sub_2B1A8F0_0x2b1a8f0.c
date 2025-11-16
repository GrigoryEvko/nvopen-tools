// Function: sub_2B1A8F0
// Address: 0x2b1a8f0
//
__int64 __fastcall sub_2B1A8F0(__int64 a1, __int64 a2)
{
  unsigned int v2; // r14d
  __int64 v3; // rdx
  unsigned int v4; // r13d
  unsigned int *v5; // rax
  int v6; // ecx
  unsigned __int64 v7; // r10
  unsigned __int64 v8; // r10
  unsigned __int64 *v9; // rdx
  unsigned __int64 *v10; // r15
  unsigned __int64 v11; // rdx
  unsigned int *v13; // rax
  __int64 *v14; // rbx
  unsigned __int64 v15; // [rsp+0h] [rbp-80h]
  int v16; // [rsp+8h] [rbp-78h]
  int v17; // [rsp+Ch] [rbp-74h]
  unsigned __int64 *v18; // [rsp+10h] [rbp-70h] BYREF
  unsigned int v19; // [rsp+18h] [rbp-68h]
  unsigned __int64 *v20; // [rsp+20h] [rbp-60h] BYREF
  unsigned int v21; // [rsp+28h] [rbp-58h]
  unsigned __int64 v22; // [rsp+30h] [rbp-50h] BYREF
  unsigned int v23; // [rsp+38h] [rbp-48h]
  unsigned __int64 v24; // [rsp+40h] [rbp-40h]
  unsigned int v25; // [rsp+48h] [rbp-38h]

  v2 = 1;
  if ( *(_BYTE *)a2 == 13 )
    return v2;
  if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
    v3 = *(_QWORD *)(a2 - 8);
  else
    v3 = a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF);
  sub_9AC3E0((__int64)&v22, *(_QWORD *)(v3 + 32), *(_QWORD *)(*(_QWORD *)a1 + 3344LL), 0, 0, 0, 0, 1);
  v4 = v23;
  v17 = **(_DWORD **)(a1 + 8);
  v5 = *(unsigned int **)(a1 + 16);
  v6 = *v5;
  v21 = v23;
  v16 = v6;
  if ( v23 <= 0x40 )
  {
    v7 = v22;
LABEL_6:
    v19 = v4;
    v8 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v4) & ~v7;
    v9 = 0;
    if ( v4 )
      v9 = (unsigned __int64 *)v8;
    v18 = v9;
    v10 = v9;
    v11 = *v5;
LABEL_9:
    v2 = 0;
    if ( (unsigned __int64)v10 >= v11 )
      goto LABEL_10;
    goto LABEL_22;
  }
  sub_C43780((__int64)&v20, (const void **)&v22);
  v4 = v21;
  if ( v21 <= 0x40 )
  {
    v5 = *(unsigned int **)(a1 + 16);
    v7 = (unsigned __int64)v20;
    goto LABEL_6;
  }
  sub_C43D10((__int64)&v20);
  v4 = v21;
  v10 = v20;
  v13 = *(unsigned int **)(a1 + 16);
  v19 = v21;
  v18 = v20;
  v11 = *v13;
  if ( v21 <= 0x40 )
    goto LABEL_9;
  v15 = *v13;
  if ( v4 - (unsigned int)sub_C444A0((__int64)&v18) > 0x40 )
  {
    v2 = 0;
    goto LABEL_27;
  }
  if ( v15 > *v10 )
  {
LABEL_22:
    if ( (*(_BYTE *)(a2 + 7) & 0x40) != 0 )
      v14 = *(__int64 **)(a2 - 8);
    else
      v14 = (__int64 *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
    v2 = v17 - v16;
    LOBYTE(v2) = (unsigned int)sub_9AF8B0(
                                 *v14,
                                 *(_QWORD *)(*(_QWORD *)a1 + 3344LL),
                                 0,
                                 *(_QWORD *)(*(_QWORD *)a1 + 3328LL),
                                 0,
                                 *(_QWORD *)(*(_QWORD *)a1 + 3320LL),
                                 1) > v17 - v16;
    if ( v4 <= 0x40 )
      goto LABEL_10;
LABEL_27:
    if ( !v10 )
      goto LABEL_10;
    goto LABEL_30;
  }
  v2 = 0;
LABEL_30:
  j_j___libc_free_0_0((unsigned __int64)v10);
LABEL_10:
  if ( v25 > 0x40 && v24 )
    j_j___libc_free_0_0(v24);
  if ( v23 > 0x40 && v22 )
    j_j___libc_free_0_0(v22);
  return v2;
}
