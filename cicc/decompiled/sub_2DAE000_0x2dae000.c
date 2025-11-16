// Function: sub_2DAE000
// Address: 0x2dae000
//
unsigned __int64 __fastcall sub_2DAE000(__int64 a1, __int64 a2, __int64 a3)
{
  const void *v4; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  __int64 *v7; // r12
  __int64 v8; // rax
  int v9; // r15d
  __int64 v10; // rax
  __int64 v11; // r9
  __int64 v12; // rcx
  __int64 v13; // rdx
  unsigned __int64 v14; // rdi
  int v15; // ecx
  unsigned __int64 v16; // rax
  unsigned int v17; // r14d
  unsigned __int64 v18; // r8
  int v19; // ecx
  int v20; // ecx
  unsigned __int64 result; // rax
  int v22; // r15d
  int v23; // ecx
  unsigned __int64 v24; // r13
  int v25; // r15d
  __int64 v26; // [rsp+0h] [rbp-40h]
  __int64 v27; // [rsp+0h] [rbp-40h]
  __int64 v28; // [rsp+8h] [rbp-38h]
  unsigned __int64 v29; // [rsp+8h] [rbp-38h]

  v4 = (const void *)(a1 + 120);
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 8) = a3;
  *(_QWORD *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = 0;
  *(_QWORD *)(a1 + 40) = 0;
  *(_QWORD *)(a1 + 48) = 0;
  *(_QWORD *)(a1 + 56) = 0;
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_QWORD *)(a1 + 80) = 0;
  *(_QWORD *)(a1 + 88) = 0;
  *(_QWORD *)(a1 + 96) = 0;
  *(_QWORD *)(a1 + 32) = 8;
  v5 = sub_22077B0(0x40u);
  v6 = *(_QWORD *)(a1 + 32);
  *(_QWORD *)(a1 + 24) = v5;
  v7 = (__int64 *)(v5 + ((4 * v6 - 4) & 0xFFFFFFFFFFFFFFF8LL));
  v8 = sub_22077B0(0x200u);
  *(_QWORD *)(a1 + 64) = v7;
  *v7 = v8;
  *(_QWORD *)(a1 + 48) = v8;
  *(_QWORD *)(a1 + 96) = v7;
  *(_QWORD *)(a1 + 80) = v8;
  *(_QWORD *)(a1 + 40) = v8;
  *(_QWORD *)(a1 + 72) = v8;
  *(_QWORD *)(a1 + 56) = v8 + 512;
  *(_QWORD *)(a1 + 88) = v8 + 512;
  *(_QWORD *)(a1 + 104) = a1 + 120;
  *(_QWORD *)(a1 + 112) = 0x600000000LL;
  *(_DWORD *)(a1 + 168) = 0;
  *(_QWORD *)(a1 + 176) = a1 + 192;
  *(_QWORD *)(a1 + 184) = 0x600000000LL;
  *(_DWORD *)(a1 + 240) = 0;
  v28 = *(unsigned int *)(a2 + 64);
  v9 = *(_DWORD *)(a2 + 64);
  v10 = sub_2207820(32 * v28);
  v12 = v10;
  if ( v10 && v28 )
  {
    v13 = 32 * v28 + v10;
    do
    {
      v10 += 32;
      *(_OWORD *)(v10 - 32) = 0;
      *(_OWORD *)(v10 - 16) = 0;
    }
    while ( v13 != v10 );
  }
  v14 = *(_QWORD *)(a1 + 16);
  *(_QWORD *)(a1 + 16) = v12;
  if ( v14 )
    j_j___libc_free_0_0(v14);
  v15 = *(_DWORD *)(a1 + 168) & 0x3F;
  if ( v15 )
    *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8LL * *(unsigned int *)(a1 + 112) - 8) &= ~(-1LL << v15);
  v16 = *(unsigned int *)(a1 + 112);
  *(_DWORD *)(a1 + 168) = v28;
  v17 = (unsigned int)(v28 + 63) >> 6;
  v18 = v17;
  if ( v17 == v16 )
  {
LABEL_12:
    LOBYTE(v19) = v28 & 0x3F;
    if ( (v28 & 0x3F) == 0 )
      goto LABEL_14;
    goto LABEL_13;
  }
  if ( v17 < v16 )
  {
    *(_DWORD *)(a1 + 112) = v17;
    goto LABEL_12;
  }
  v11 = v17 - v16;
  if ( v17 > (unsigned __int64)*(unsigned int *)(a1 + 116) )
  {
    v27 = v17 - v16;
    sub_C8D5F0(a1 + 104, v4, (unsigned int)(v28 + 63) >> 6, 8u, v17, v11);
    v16 = *(unsigned int *)(a1 + 112);
    v11 = v27;
    v18 = (unsigned int)(v28 + 63) >> 6;
  }
  if ( 8 * v11 )
  {
    v26 = v11;
    v29 = v18;
    memset((void *)(*(_QWORD *)(a1 + 104) + 8 * v16), 0, 8 * v11);
    LODWORD(v16) = *(_DWORD *)(a1 + 112);
    v11 = v26;
    v18 = v29;
  }
  v23 = *(_DWORD *)(a1 + 168);
  *(_DWORD *)(a1 + 112) = v11 + v16;
  v19 = v23 & 0x3F;
  if ( v19 )
LABEL_13:
    *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8LL * *(unsigned int *)(a1 + 112) - 8) &= ~(-1LL << v19);
LABEL_14:
  v20 = *(_DWORD *)(a1 + 240) & 0x3F;
  if ( v20 )
    *(_QWORD *)(*(_QWORD *)(a1 + 176) + 8LL * *(unsigned int *)(a1 + 184) - 8) &= ~(-1LL << v20);
  result = *(unsigned int *)(a1 + 184);
  *(_DWORD *)(a1 + 240) = v9;
  if ( v18 == result )
  {
LABEL_19:
    v22 = v9 & 0x3F;
    if ( !v22 )
      return result;
    goto LABEL_20;
  }
  if ( v18 < result )
  {
    *(_DWORD *)(a1 + 184) = v17;
    goto LABEL_19;
  }
  v24 = v18 - result;
  if ( v18 > *(unsigned int *)(a1 + 188) )
  {
    sub_C8D5F0(a1 + 176, (const void *)(a1 + 192), v18, 8u, v18, v11);
    result = *(unsigned int *)(a1 + 184);
  }
  if ( 8 * v24 )
  {
    memset((void *)(*(_QWORD *)(a1 + 176) + 8 * result), 0, 8 * v24);
    result = *(unsigned int *)(a1 + 184);
  }
  v25 = *(_DWORD *)(a1 + 240);
  result += v24;
  *(_DWORD *)(a1 + 184) = result;
  v22 = v25 & 0x3F;
  if ( v22 )
  {
LABEL_20:
    result = ~(-1LL << v22);
    *(_QWORD *)(*(_QWORD *)(a1 + 176) + 8LL * *(unsigned int *)(a1 + 184) - 8) &= result;
  }
  return result;
}
