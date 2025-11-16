// Function: sub_16F45F0
// Address: 0x16f45f0
//
__int64 __fastcall sub_16F45F0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5)
{
  unsigned int v8; // r12d
  int v9; // eax
  int v10; // eax
  __int64 v11; // rsi
  __int64 v12; // rcx
  __int64 v14; // r13
  unsigned __int64 v15; // rax
  unsigned int v16; // edi
  __int64 v17; // rdx
  __int64 v18; // rcx
  int v19; // r8d
  __int64 v20; // rcx
  int v21; // r8d
  unsigned __int8 **v22; // r12
  unsigned __int8 **v23; // r14
  unsigned __int8 **v24; // r15
  char *v25; // rdi
  bool v26; // al
  unsigned __int8 **v27[4]; // [rsp+20h] [rbp-70h] BYREF
  __int64 v28[2]; // [rsp+40h] [rbp-50h] BYREF
  __int64 v29; // [rsp+50h] [rbp-40h] BYREF

  v8 = *(_DWORD *)(a1 + 24);
  v9 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v10 = v9 + 1;
  v11 = (unsigned int)(4 * v10);
  if ( (unsigned int)v11 >= 3 * v8 )
  {
    v14 = *(_QWORD *)(a1 + 8);
    v15 = 2 * v8 - 1;
  }
  else
  {
    v12 = v8 >> 3;
    if ( v8 - *(_DWORD *)(a1 + 20) - v10 > (unsigned int)v12 )
      goto LABEL_3;
    v14 = *(_QWORD *)(a1 + 8);
    v15 = v8 - 1;
  }
  v16 = (((((((((v15 | (v15 >> 1)) >> 2) | v15 | (v15 >> 1)) >> 4) | ((v15 | (v15 >> 1)) >> 2) | v15 | (v15 >> 1)) >> 8)
         | ((((v15 | (v15 >> 1)) >> 2) | v15 | (v15 >> 1)) >> 4)
         | ((v15 | (v15 >> 1)) >> 2)
         | v15
         | (v15 >> 1)) >> 16)
       | ((((((v15 | (v15 >> 1)) >> 2) | v15 | (v15 >> 1)) >> 4) | ((v15 | (v15 >> 1)) >> 2) | v15 | (v15 >> 1)) >> 8)
       | ((((v15 | (v15 >> 1)) >> 2) | v15 | (v15 >> 1)) >> 4)
       | ((v15 | (v15 >> 1)) >> 2)
       | v15
       | (v15 >> 1))
      + 1;
  if ( v16 < 0x40 )
    v16 = 64;
  *(_DWORD *)(a1 + 24) = v16;
  *(_QWORD *)(a1 + 8) = sub_22077B0((unsigned __int64)v16 << 6);
  if ( v14 )
  {
    sub_16F3FB0(a1, v14, v14 + ((unsigned __int64)v8 << 6), v18, v19);
    j___libc_free_0(v14);
  }
  else
  {
    sub_16F2B70(a1, v11, v17, v18, v19);
  }
  sub_16F3520(a1, a2, v28, v20, v21);
  a3 = v28[0];
  v10 = *(_DWORD *)(a1 + 16) + 1;
LABEL_3:
  *(_DWORD *)(a1 + 16) = v10;
  if ( (unsigned __int8)sub_16F23B0((unsigned __int8 *)0xFFFFFFFFFFFFFFFFLL, 0, 0, v12, a5) )
  {
    if ( *(_QWORD *)(a3 + 8) != -1 )
      --*(_DWORD *)(a1 + 20);
    return a3;
  }
  sub_16F2420(v28, (unsigned __int8 *)0xFFFFFFFFFFFFFFFFLL, 0);
  sub_16F25B0(v27, (__int64)v28);
  v22 = v27[0];
  v23 = v27[1];
  v24 = v27[2];
  if ( (__int64 *)v28[0] != &v29 )
    j_j___libc_free_0(v28[0], v29 + 1);
  v25 = *(char **)(a3 + 8);
  if ( v23 == (unsigned __int8 **)-1LL )
  {
    v26 = v25 + 1 == 0;
  }
  else
  {
    v26 = v25 + 2 == 0;
    if ( v23 != (unsigned __int8 **)-2LL )
    {
      if ( *(unsigned __int8 ***)(a3 + 16) != v24 || v24 && memcmp(v25, v23, (size_t)v24) )
        goto LABEL_18;
      goto LABEL_19;
    }
  }
  if ( !v26 )
LABEL_18:
    --*(_DWORD *)(a1 + 20);
LABEL_19:
  if ( v22 )
  {
    if ( *v22 != (unsigned __int8 *)(v22 + 2) )
      j_j___libc_free_0(*v22, v22[2] + 1);
    j_j___libc_free_0(v22, 32);
  }
  return a3;
}
