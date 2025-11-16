// Function: sub_C6D9C0
// Address: 0xc6d9c0
//
__int64 __fastcall sub_C6D9C0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r12
  int v5; // eax
  int v6; // eax
  __int64 v7; // r12
  __int64 *v8; // r13
  char *v9; // rdi
  bool v10; // al
  __int64 *v12; // r14
  __int64 *v13; // r15
  size_t v14; // r14
  unsigned __int64 v15; // rax
  unsigned int v16; // edi
  __int64 v17; // r12
  _QWORD v18[5]; // [rsp+8h] [rbp-98h] BYREF
  __int64 *v19[4]; // [rsp+30h] [rbp-70h] BYREF
  __int64 v20[2]; // [rsp+50h] [rbp-50h] BYREF
  __int64 v21; // [rsp+60h] [rbp-40h] BYREF

  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_DWORD *)(a1 + 16);
  v18[0] = a3;
  ++*(_QWORD *)a1;
  v6 = v5 + 1;
  if ( 4 * v6 >= (unsigned int)(3 * v4) )
  {
    v14 = *(_QWORD *)(a1 + 8);
    v15 = (unsigned int)(2 * v4 - 1);
  }
  else
  {
    if ( (int)v4 - *(_DWORD *)(a1 + 20) - v6 > (unsigned int)v4 >> 3 )
      goto LABEL_3;
    v14 = *(_QWORD *)(a1 + 8);
    v15 = (unsigned int)(v4 - 1);
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
  *(_QWORD *)(a1 + 8) = sub_C7D670((unsigned __int64)v16 << 6, 8);
  if ( v14 )
  {
    v17 = v4 << 6;
    sub_C6D3C0(a1, v14, v14 + v17);
    sub_C7D6A0(v14, v17, 8);
  }
  else
  {
    sub_C6BD30(a1);
  }
  sub_C6BF30(a1, a2, v18);
  v6 = *(_DWORD *)(a1 + 16) + 1;
LABEL_3:
  *(_DWORD *)(a1 + 16) = v6;
  v18[2] = -1;
  v18[3] = 0;
  if ( (unsigned __int8)sub_C6A630((char *)0xFFFFFFFFFFFFFFFFLL, 0, 0) )
  {
    v7 = v18[0];
    v8 = 0;
    v9 = *(char **)(v18[0] + 8LL);
LABEL_5:
    v10 = v9 + 1 == 0;
    goto LABEL_6;
  }
  sub_C6B0E0(v20, -1, 0);
  sub_C6B270(v19, (__int64)v20);
  v8 = v19[0];
  v12 = v19[1];
  v13 = v19[2];
  if ( (__int64 *)v20[0] != &v21 )
    j_j___libc_free_0(v20[0], v21 + 1);
  v7 = v18[0];
  v9 = *(char **)(v18[0] + 8LL);
  if ( v12 == (__int64 *)-1LL )
    goto LABEL_5;
  v10 = v9 + 2 == 0;
  if ( v12 == (__int64 *)-2LL )
  {
LABEL_6:
    if ( v10 )
      goto LABEL_7;
    goto LABEL_12;
  }
  if ( *(__int64 **)(v18[0] + 16LL) != v13 )
  {
LABEL_12:
    --*(_DWORD *)(a1 + 20);
    goto LABEL_7;
  }
  if ( v13 )
  {
    v10 = memcmp(v9, v12, (size_t)v13) == 0;
    goto LABEL_6;
  }
LABEL_7:
  if ( v8 )
  {
    if ( (__int64 *)*v8 != v8 + 2 )
      j_j___libc_free_0(*v8, v8[2] + 1);
    j_j___libc_free_0(v8, 32);
  }
  return v7;
}
