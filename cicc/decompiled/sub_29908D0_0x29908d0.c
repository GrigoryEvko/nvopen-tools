// Function: sub_29908D0
// Address: 0x29908d0
//
__int64 __fastcall sub_29908D0(__int64 a1, __int64 *a2)
{
  __int64 v4; // rdx
  unsigned int v5; // esi
  __int64 v6; // r8
  __int64 v7; // rdi
  int v8; // r10d
  __int64 v9; // r13
  unsigned int v10; // ecx
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // rax
  int v15; // eax
  int v16; // ecx
  char *v17; // r14
  __int64 v18; // rdx
  unsigned __int64 v19; // rcx
  __int64 v20; // rdx
  unsigned __int64 v21; // rsi
  __int64 v22; // rsi
  _QWORD *v23; // rdi
  _BYTE *v24; // rdi
  unsigned __int64 v25; // r15
  __int64 v26; // rdi
  __int64 v27; // [rsp+0h] [rbp-B0h] BYREF
  int v28; // [rsp+8h] [rbp-A8h]
  char *v29; // [rsp+10h] [rbp-A0h]
  __int64 v30; // [rsp+18h] [rbp-98h]
  char v31; // [rsp+20h] [rbp-90h] BYREF
  __int64 v32; // [rsp+40h] [rbp-70h] BYREF
  _BYTE *v33; // [rsp+48h] [rbp-68h]
  __int64 v34; // [rsp+50h] [rbp-60h]
  _BYTE v35[88]; // [rsp+58h] [rbp-58h] BYREF

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  v28 = 0;
  v27 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    v32 = 0;
LABEL_27:
    v17 = (char *)&v32;
    sub_29906F0(a1, 2 * v5);
LABEL_28:
    sub_298C180(a1, &v27, &v32);
    v4 = v27;
    v9 = v32;
    v16 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_15;
  }
  v6 = v5 - 1;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 1;
  v9 = 0;
  v10 = v6 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
  v11 = v7 + 16LL * v10;
  v12 = *(_QWORD *)v11;
  if ( v4 == *(_QWORD *)v11 )
  {
LABEL_3:
    v13 = *(unsigned int *)(v11 + 8);
    return *(_QWORD *)(a1 + 32) + 56 * v13 + 8;
  }
  while ( v12 != -4096 )
  {
    if ( !v9 && v12 == -8192 )
      v9 = v11;
    v10 = v6 & (v8 + v10);
    v11 = v7 + 16LL * v10;
    v12 = *(_QWORD *)v11;
    if ( v4 == *(_QWORD *)v11 )
      goto LABEL_3;
    ++v8;
  }
  if ( !v9 )
    v9 = v11;
  v15 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v16 = v15 + 1;
  v32 = v9;
  if ( 4 * (v15 + 1) >= 3 * v5 )
    goto LABEL_27;
  v17 = (char *)&v32;
  if ( v5 - *(_DWORD *)(a1 + 20) - v16 <= v5 >> 3 )
  {
    sub_29906F0(a1, v5);
    goto LABEL_28;
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v16;
  if ( *(_QWORD *)v9 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v9 = v4;
  *(_DWORD *)(v9 + 8) = v28;
  v18 = *a2;
  v19 = *(unsigned int *)(a1 + 44);
  v29 = &v31;
  v32 = v18;
  v20 = *(unsigned int *)(a1 + 40);
  v30 = 0x200000000LL;
  v21 = v20 + 1;
  v34 = 0x200000000LL;
  v13 = v20;
  v33 = v35;
  if ( v20 + 1 > v19 )
  {
    v25 = *(_QWORD *)(a1 + 32);
    v26 = a1 + 32;
    if ( v25 > (unsigned __int64)&v32 || (unsigned __int64)&v32 >= v25 + 56 * v20 )
    {
      sub_298C290(v26, v21, v20, v19, v6, v12);
      v20 = *(unsigned int *)(a1 + 40);
      v22 = *(_QWORD *)(a1 + 32);
      v13 = v20;
    }
    else
    {
      sub_298C290(v26, v21, v20, v19, v6, v12);
      v22 = *(_QWORD *)(a1 + 32);
      v20 = *(unsigned int *)(a1 + 40);
      v17 = (char *)&v32 + v22 - v25;
      v13 = v20;
    }
  }
  else
  {
    v22 = *(_QWORD *)(a1 + 32);
  }
  v23 = (_QWORD *)(v22 + 56 * v20);
  if ( v23 )
  {
    *v23 = *(_QWORD *)v17;
    v23[1] = v23 + 3;
    v23[2] = 0x200000000LL;
    if ( *((_DWORD *)v17 + 4) )
      sub_2988720((__int64)(v23 + 1), (__int64)(v17 + 8), v20, 7 * v20, v6, v12);
    v13 = *(unsigned int *)(a1 + 40);
  }
  v24 = v33;
  *(_DWORD *)(a1 + 40) = v13 + 1;
  if ( v24 != v35 )
  {
    _libc_free((unsigned __int64)v24);
    v13 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  }
  *(_DWORD *)(v9 + 8) = v13;
  return *(_QWORD *)(a1 + 32) + 56 * v13 + 8;
}
