// Function: sub_2909AD0
// Address: 0x2909ad0
//
__int64 __fastcall sub_2909AD0(__int64 a1, __int64 *a2)
{
  __int64 v4; // rdx
  unsigned int v5; // esi
  __int64 v6; // r8
  __int64 v7; // rdi
  int v8; // r10d
  __int64 v9; // r12
  unsigned int v10; // ecx
  __int64 v11; // rax
  __int64 v12; // r9
  __int64 v13; // rax
  int v15; // eax
  __int64 v16; // rcx
  __int64 *v17; // r14
  __int64 v18; // rax
  __int64 v19; // r8
  __int64 v20; // r9
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rcx
  unsigned __int64 v23; // rsi
  int v24; // eax
  __int64 v25; // rsi
  __int64 v26; // rdi
  __int64 v27; // rdi
  unsigned __int64 v28; // rax
  __int64 v29; // rdi
  __int64 v30; // [rsp+8h] [rbp-B8h]
  __int64 v31; // [rsp+10h] [rbp-B0h] BYREF
  int v32; // [rsp+18h] [rbp-A8h]
  _QWORD v33[6]; // [rsp+20h] [rbp-A0h] BYREF
  __int64 v34; // [rsp+50h] [rbp-70h] BYREF
  _BYTE v35[104]; // [rsp+58h] [rbp-68h] BYREF

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  v32 = 0;
  v31 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    v34 = 0;
LABEL_23:
    v17 = &v34;
    sub_B23080(a1, 2 * v5);
LABEL_24:
    sub_B1C700(a1, &v31, &v34);
    v4 = v31;
    v9 = v34;
    v16 = (unsigned int)(*(_DWORD *)(a1 + 16) + 1);
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
  v16 = (unsigned int)(v15 + 1);
  v34 = v9;
  if ( 4 * (int)v16 >= 3 * v5 )
    goto LABEL_23;
  v17 = &v34;
  if ( v5 - *(_DWORD *)(a1 + 20) - (unsigned int)v16 <= v5 >> 3 )
  {
    sub_B23080(a1, v5);
    goto LABEL_24;
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v16;
  if ( *(_QWORD *)v9 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v9 = v4;
  v33[0] = 0;
  *(_DWORD *)(v9 + 8) = v32;
  v18 = *a2;
  v33[4] = &v34;
  v34 = v18;
  memset(&v33[1], 0, 24);
  v33[5] = 0;
  sub_2900F20((__int64)v35, (__int64)v33, v4, v16, v6, v12);
  v21 = *(unsigned int *)(a1 + 40);
  v22 = *(unsigned int *)(a1 + 44);
  v23 = v21 + 1;
  v24 = *(_DWORD *)(a1 + 40);
  if ( v21 + 1 > v22 )
  {
    v28 = *(_QWORD *)(a1 + 32);
    v29 = a1 + 32;
    if ( v28 > (unsigned __int64)&v34
      || (v30 = *(_QWORD *)(a1 + 32), v22 = 7 * v21, v21 = v28 + 56 * v21, (unsigned __int64)&v34 >= v21) )
    {
      sub_2904460(v29, v23, v21, v22, v19, v20);
      v21 = *(unsigned int *)(a1 + 40);
      v25 = *(_QWORD *)(a1 + 32);
      v24 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_2904460(v29, v23, v21, v22, v19, v20);
      v25 = *(_QWORD *)(a1 + 32);
      v21 = *(unsigned int *)(a1 + 40);
      v17 = (__int64 *)&v35[v25 - 8 - v30];
      v24 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v25 = *(_QWORD *)(a1 + 32);
  }
  v26 = v25 + 56 * v21;
  if ( v26 )
  {
    v27 = v26 + 8;
    *(_QWORD *)(v27 - 8) = *v17;
    sub_2900F20(v27, (__int64)(v17 + 1), v21, 7 * v21, v19, v20);
    v24 = *(_DWORD *)(a1 + 40);
  }
  *(_DWORD *)(a1 + 40) = v24 + 1;
  sub_25FC810((__int64)v35);
  sub_25FC810((__int64)v33);
  v13 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  *(_DWORD *)(v9 + 8) = v13;
  return *(_QWORD *)(a1 + 32) + 56 * v13 + 8;
}
