// Function: sub_34415F0
// Address: 0x34415f0
//
__int64 __fastcall sub_34415F0(__int64 *a1, __int64 a2, __m128i a3)
{
  __int64 v3; // rax
  unsigned int v4; // r12d
  const void **v5; // r13
  int v6; // r8d
  __int64 result; // rax
  __int64 v8; // rbx
  unsigned int v9; // eax
  __int64 v10; // r14
  __int64 v11; // r12
  unsigned __int8 *v12; // rax
  unsigned __int8 *v13; // rdx
  unsigned __int8 *v14; // r15
  __int64 v15; // rdx
  unsigned __int8 *v16; // r14
  unsigned __int8 **v17; // rdx
  __int64 v18; // r12
  unsigned __int8 *v19; // rax
  __int64 v20; // r9
  unsigned __int8 *v21; // rdx
  unsigned __int8 *v22; // r15
  __int64 v23; // rdx
  unsigned __int8 *v24; // r14
  unsigned __int8 **v25; // rdx
  unsigned int v27; // edx
  unsigned __int64 v29; // rax
  __int64 v30; // [rsp-8h] [rbp-58h]
  unsigned __int64 v31; // [rsp+0h] [rbp-50h] BYREF
  unsigned int v32; // [rsp+8h] [rbp-48h]
  unsigned __int64 v33; // [rsp+10h] [rbp-40h] BYREF
  unsigned int v34; // [rsp+18h] [rbp-38h]

  v3 = *(_QWORD *)(*(_QWORD *)a2 + 96LL);
  v4 = *(_DWORD *)(v3 + 32);
  if ( v4 <= 0x40 )
  {
    if ( !*(_QWORD *)(v3 + 24) )
      return 0;
    _RAX = *(_QWORD *)(v3 + 24);
    v8 = *a1;
    v32 = v4;
    v31 = _RAX;
    goto LABEL_19;
  }
  v5 = (const void **)(v3 + 24);
  v6 = sub_C444A0(v3 + 24);
  result = 0;
  if ( v4 == v6 )
    return result;
  v32 = v4;
  v8 = *a1;
  sub_C43780((__int64)&v31, v5);
  v4 = v32;
  if ( v32 <= 0x40 )
  {
    _RAX = v31;
LABEL_19:
    v27 = 64;
    LODWORD(v10) = v4;
    __asm { tzcnt   rcx, rax }
    if ( _RAX )
      v27 = _RCX;
    if ( v27 <= v4 )
      LODWORD(v10) = v27;
    if ( !(_DWORD)v10 )
      goto LABEL_5;
    v29 = _RAX >> v10;
    if ( v27 >= v4 )
      v29 = 0;
    v31 = v29;
LABEL_27:
    v10 = (unsigned int)v10;
    **(_BYTE **)v8 = 1;
    goto LABEL_6;
  }
  v9 = sub_C44590((__int64)&v31);
  LODWORD(v10) = v9;
  if ( v9 )
  {
    sub_C482E0((__int64)&v31, v9);
    goto LABEL_27;
  }
LABEL_5:
  v10 = 0;
LABEL_6:
  sub_C473B0((__int64)&v33, (__int64)&v31);
  v11 = *(_QWORD *)(v8 + 8);
  v12 = sub_3400BD0(
          *(_QWORD *)(v8 + 16),
          v10,
          *(_QWORD *)(v8 + 24),
          **(unsigned int **)(v8 + 32),
          *(_QWORD *)(*(_QWORD *)(v8 + 32) + 8LL),
          0,
          a3,
          0);
  v14 = v13;
  v15 = *(unsigned int *)(v11 + 8);
  v16 = v12;
  if ( v15 + 1 > (unsigned __int64)*(unsigned int *)(v11 + 12) )
  {
    sub_C8D5F0(v11, (const void *)(v11 + 16), v15 + 1, 0x10u, v15 + 1, v30);
    v15 = *(unsigned int *)(v11 + 8);
  }
  v17 = (unsigned __int8 **)(*(_QWORD *)v11 + 16 * v15);
  *v17 = v16;
  v17[1] = v14;
  ++*(_DWORD *)(v11 + 8);
  v18 = *(_QWORD *)(v8 + 40);
  v19 = sub_34007B0(
          *(_QWORD *)(v8 + 16),
          (__int64)&v33,
          *(_QWORD *)(v8 + 24),
          **(_DWORD **)(v8 + 48),
          *(_QWORD *)(*(_QWORD *)(v8 + 48) + 8LL),
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
  if ( v34 > 0x40 && v33 )
    j_j___libc_free_0_0(v33);
  if ( v32 > 0x40 )
  {
    if ( v31 )
      j_j___libc_free_0_0(v31);
  }
  return 1;
}
