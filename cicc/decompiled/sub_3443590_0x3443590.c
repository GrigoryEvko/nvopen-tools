// Function: sub_3443590
// Address: 0x3443590
//
__int64 __fastcall sub_3443590(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v3; // r12
  unsigned int v4; // r14d
  int v5; // r8d
  __int64 result; // rax
  __int64 v7; // rbx
  unsigned int v8; // edx
  __int64 v9; // r15
  unsigned __int64 v10; // rdx
  __int64 v11; // r12
  __int64 v12; // r13
  __int64 v13; // rdx
  unsigned __int8 *v14; // r8
  __int64 v15; // rax
  __int64 v16; // r9
  unsigned __int8 **v17; // rax
  __int64 v18; // r13
  unsigned __int8 *v19; // rax
  __int64 v20; // r9
  unsigned __int8 *v21; // rdx
  unsigned __int8 *v22; // r15
  __int64 v23; // rdx
  unsigned __int8 *v24; // r14
  unsigned __int8 **v25; // rdx
  __int64 v26; // r13
  unsigned __int8 *v27; // rax
  unsigned __int8 *v28; // rdx
  unsigned __int8 *v29; // r15
  __int64 v30; // rdx
  unsigned __int8 *v31; // r14
  unsigned __int8 **v32; // rdx
  __int64 v33; // r13
  unsigned __int8 *v34; // rax
  __int64 v35; // r9
  unsigned __int8 *v36; // rdx
  unsigned __int8 *v37; // r15
  __int64 v38; // rdx
  unsigned __int8 *v39; // r14
  unsigned __int8 **v40; // rdx
  __int64 v41; // rax
  char v42; // cl
  unsigned int v43; // r12d
  __int64 v44; // rax
  __int64 v45; // [rsp-8h] [rbp-68h]
  int v46; // [rsp+0h] [rbp-60h]
  unsigned __int8 *v47; // [rsp+0h] [rbp-60h]
  __int64 v48; // [rsp+8h] [rbp-58h]
  unsigned __int64 v49; // [rsp+10h] [rbp-50h] BYREF
  unsigned int v50; // [rsp+18h] [rbp-48h]
  unsigned int v51; // [rsp+20h] [rbp-40h]

  v3 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v4 = *(_DWORD *)(v3 + 32);
  if ( v4 <= 0x40 )
  {
    result = 0;
    if ( !*(_QWORD *)(v3 + 24) )
      return result;
  }
  else
  {
    v5 = sub_C444A0(v3 + 24);
    result = 0;
    if ( v4 == v5 )
      return result;
  }
  v7 = *a1;
  sub_3719380(&v49, v3 + 24);
  v8 = *(_DWORD *)(v3 + 32);
  if ( v8 > 0x40 )
  {
    v46 = *(_DWORD *)(v3 + 32);
    LODWORD(v9) = v46 - 1;
    if ( (unsigned int)sub_C444A0(v3 + 24) != v46 - 1 && v46 != (unsigned int)sub_C445E0(v3 + 24) )
    {
      if ( (*(_QWORD *)(*(_QWORD *)(v3 + 24) + 8LL * ((unsigned int)v9 >> 6)) & (1LL << v9)) == 0 )
      {
        if ( v46 != (unsigned int)sub_C444A0(v3 + 24) )
        {
LABEL_10:
          v10 = v49;
          if ( v50 > 0x40 )
            v10 = *(_QWORD *)(v49 + 8LL * ((v50 - 1) >> 6));
          v11 = -1;
          v9 = (v10 & (1LL << ((unsigned __int8)v50 - 1))) != 0;
          goto LABEL_17;
        }
        goto LABEL_29;
      }
      goto LABEL_34;
    }
    v9 = **(int **)(v3 + 24);
LABEL_14:
    if ( v50 > 0x40 )
    {
      *(_QWORD *)v49 = 0;
      memset((void *)(v49 + 8), 0, 8 * (unsigned int)(((unsigned __int64)v50 + 63) >> 6) - 8);
    }
    else
    {
      v49 = 0;
    }
    v51 = 0;
    v11 = 0;
    goto LABEL_17;
  }
  v41 = *(_QWORD *)(v3 + 24);
  if ( v41 == 1 )
  {
    if ( !v8 )
    {
      v9 = 0;
      goto LABEL_14;
    }
    v42 = 64 - v8;
    goto LABEL_42;
  }
  v9 = 0;
  if ( !v8 )
    goto LABEL_14;
  v42 = 64 - v8;
  if ( v41 == 0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v8) )
  {
LABEL_42:
    v9 = (int)(v41 << v42 >> v42);
    goto LABEL_14;
  }
  if ( !_bittest64(&v41, v8 - 1) )
  {
    if ( v41 )
      goto LABEL_10;
    goto LABEL_38;
  }
LABEL_34:
  v43 = v50;
  v44 = 1LL << ((unsigned __int8)v50 - 1);
  if ( v50 <= 0x40 )
  {
    if ( (v44 & v49) == 0 )
    {
      LOBYTE(v9) = v49 == 0;
      goto LABEL_37;
    }
  }
  else if ( (*(_QWORD *)(v49 + 8LL * ((v50 - 1) >> 6)) & v44) == 0 )
  {
    LOBYTE(v9) = v43 == (unsigned int)sub_C444A0((__int64)&v49);
LABEL_37:
    v9 = (__int64)((unsigned __int64)((unsigned int)v9 ^ 1) << 63) >> 63;
LABEL_38:
    v11 = -1;
    goto LABEL_17;
  }
LABEL_29:
  v11 = -1;
  v9 = 0;
LABEL_17:
  v12 = *(_QWORD *)v7;
  v14 = sub_34007B0(
          *(_QWORD *)(v7 + 8),
          (__int64)&v49,
          *(_QWORD *)(v7 + 16),
          **(_DWORD **)(v7 + 24),
          *(_QWORD *)(*(_QWORD *)(v7 + 24) + 8LL),
          0,
          a3,
          0);
  v15 = *(unsigned int *)(v12 + 8);
  v16 = v13;
  if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(v12 + 12) )
  {
    v47 = v14;
    v48 = v13;
    sub_C8D5F0(v12, (const void *)(v12 + 16), v15 + 1, 0x10u, (__int64)v14, v13);
    v15 = *(unsigned int *)(v12 + 8);
    v14 = v47;
    v16 = v48;
  }
  v17 = (unsigned __int8 **)(*(_QWORD *)v12 + 16 * v15);
  *v17 = v14;
  v17[1] = (unsigned __int8 *)v16;
  ++*(_DWORD *)(v12 + 8);
  v18 = *(_QWORD *)(v7 + 32);
  v19 = sub_3401400(
          *(_QWORD *)(v7 + 8),
          v9,
          *(_QWORD *)(v7 + 16),
          **(unsigned int **)(v7 + 24),
          *(_QWORD *)(*(_QWORD *)(v7 + 24) + 8LL),
          0,
          a3,
          0);
  v22 = v21;
  v23 = *(unsigned int *)(v18 + 8);
  v24 = v19;
  if ( v23 + 1 > (unsigned __int64)*(unsigned int *)(v18 + 12) )
  {
    sub_C8D5F0(v18, (const void *)(v18 + 16), v23 + 1, 0x10u, v23 + 1, v20);
    v23 = *(unsigned int *)(v18 + 8);
  }
  v25 = (unsigned __int8 **)(*(_QWORD *)v18 + 16 * v23);
  *v25 = v24;
  v25[1] = v22;
  ++*(_DWORD *)(v18 + 8);
  v26 = *(_QWORD *)(v7 + 40);
  v27 = sub_3400BD0(
          *(_QWORD *)(v7 + 8),
          v51,
          *(_QWORD *)(v7 + 16),
          **(unsigned int **)(v7 + 48),
          *(_QWORD *)(*(_QWORD *)(v7 + 48) + 8LL),
          0,
          a3,
          0);
  v29 = v28;
  v30 = *(unsigned int *)(v26 + 8);
  v31 = v27;
  if ( v30 + 1 > (unsigned __int64)*(unsigned int *)(v26 + 12) )
  {
    sub_C8D5F0(v26, (const void *)(v26 + 16), v30 + 1, 0x10u, v30 + 1, v45);
    v30 = *(unsigned int *)(v26 + 8);
  }
  v32 = (unsigned __int8 **)(*(_QWORD *)v26 + 16 * v30);
  *v32 = v31;
  v32[1] = v29;
  ++*(_DWORD *)(v26 + 8);
  v33 = *(_QWORD *)(v7 + 56);
  v34 = sub_3401400(
          *(_QWORD *)(v7 + 8),
          v11,
          *(_QWORD *)(v7 + 16),
          **(unsigned int **)(v7 + 24),
          *(_QWORD *)(*(_QWORD *)(v7 + 24) + 8LL),
          0,
          a3,
          0);
  v37 = v36;
  v38 = *(unsigned int *)(v33 + 8);
  v39 = v34;
  if ( v38 + 1 > (unsigned __int64)*(unsigned int *)(v33 + 12) )
  {
    sub_C8D5F0(v33, (const void *)(v33 + 16), v38 + 1, 0x10u, v38 + 1, v35);
    v38 = *(unsigned int *)(v33 + 8);
  }
  v40 = (unsigned __int8 **)(*(_QWORD *)v33 + 16 * v38);
  *v40 = v39;
  v40[1] = v37;
  ++*(_DWORD *)(v33 + 8);
  if ( v50 > 0x40 )
  {
    if ( v49 )
      j_j___libc_free_0_0(v49);
  }
  return 1;
}
