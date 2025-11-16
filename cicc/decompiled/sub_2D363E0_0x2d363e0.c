// Function: sub_2D363E0
// Address: 0x2d363e0
//
__int64 __fastcall sub_2D363E0(__int64 a1, __int64 *a2)
{
  __int64 v4; // rdx
  unsigned int v5; // esi
  __int64 v6; // r9
  __int64 v7; // r8
  __int64 v8; // r13
  int v9; // r10d
  unsigned int v10; // ecx
  __int64 v11; // rax
  __int64 v12; // rdi
  __int64 v13; // rax
  int v15; // eax
  int v16; // edx
  __int64 v17; // rdx
  unsigned __int64 v18; // rcx
  __int64 v19; // rdx
  unsigned __int64 v20; // rsi
  int v21; // eax
  __int64 v22; // rdi
  char *v23; // rsi
  __int64 v24; // rdi
  unsigned __int64 v25; // r12
  __int64 v26; // rdi
  __int64 v27; // [rsp+0h] [rbp-A0h] BYREF
  int v28; // [rsp+8h] [rbp-98h]
  _QWORD v29[2]; // [rsp+10h] [rbp-90h] BYREF
  char v30; // [rsp+20h] [rbp-80h] BYREF
  __int64 v31; // [rsp+40h] [rbp-60h] BYREF
  _QWORD v32[2]; // [rsp+48h] [rbp-58h] BYREF
  char v33; // [rsp+58h] [rbp-48h] BYREF

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  v28 = 0;
  v27 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    v31 = 0;
LABEL_25:
    sub_2D36220(a1, 2 * v5);
LABEL_26:
    sub_2D2BD40(a1, &v27, &v31);
    v8 = v31;
    v16 = *(_DWORD *)(a1 + 16) + 1;
    goto LABEL_15;
  }
  v6 = v5 - 1;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = 0;
  v9 = 1;
  v10 = v6 & (37 * v4);
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
    if ( !v8 && v12 == -8192 )
      v8 = v11;
    v10 = v6 & (v9 + v10);
    v11 = v7 + 16LL * v10;
    v12 = *(_QWORD *)v11;
    if ( v4 == *(_QWORD *)v11 )
      goto LABEL_3;
    ++v9;
  }
  if ( !v8 )
    v8 = v11;
  v15 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  v16 = v15 + 1;
  v31 = v8;
  if ( 4 * (v15 + 1) >= 3 * v5 )
    goto LABEL_25;
  if ( v5 - *(_DWORD *)(a1 + 20) - v16 <= v5 >> 3 )
  {
    sub_2D36220(a1, v5);
    goto LABEL_26;
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v16;
  if ( *(_QWORD *)v8 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v8 = v27;
  *(_DWORD *)(v8 + 8) = v28;
  v17 = *a2;
  v18 = *(unsigned int *)(a1 + 44);
  v29[0] = &v30;
  v31 = v17;
  v32[0] = &v33;
  v19 = *(unsigned int *)(a1 + 40);
  v29[1] = 0x100000000LL;
  v20 = v19 + 1;
  v32[1] = 0x100000000LL;
  v21 = v19;
  if ( v19 + 1 > v18 )
  {
    v25 = *(_QWORD *)(a1 + 32);
    v26 = a1 + 32;
    if ( v25 > (unsigned __int64)&v31 || (unsigned __int64)&v31 >= v25 + 56 * v19 )
    {
      sub_2D2E580(v26, v20, v19, v18, v7, v6);
      v19 = *(unsigned int *)(a1 + 40);
      v22 = *(_QWORD *)(a1 + 32);
      v23 = (char *)&v31;
      v21 = *(_DWORD *)(a1 + 40);
    }
    else
    {
      sub_2D2E580(v26, v20, v19, v18, v7, v6);
      v22 = *(_QWORD *)(a1 + 32);
      v19 = *(unsigned int *)(a1 + 40);
      v23 = (char *)&v32[-1] + v22 - v25;
      v21 = *(_DWORD *)(a1 + 40);
    }
  }
  else
  {
    v22 = *(_QWORD *)(a1 + 32);
    v23 = (char *)&v31;
  }
  v24 = v22 + 56 * v19;
  if ( v24 )
  {
    *(_QWORD *)v24 = *(_QWORD *)v23;
    *(_QWORD *)(v24 + 8) = v24 + 24;
    *(_QWORD *)(v24 + 16) = 0x100000000LL;
    if ( *((_DWORD *)v23 + 4) )
      sub_2D29780((unsigned int *)(v24 + 8), (__int64)(v23 + 8), v19, 7 * v19, v7, v6);
    v21 = *(_DWORD *)(a1 + 40);
  }
  *(_DWORD *)(a1 + 40) = v21 + 1;
  sub_2D288B0((__int64)v32);
  sub_2D288B0((__int64)v29);
  v13 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  *(_DWORD *)(v8 + 8) = v13;
  return *(_QWORD *)(a1 + 32) + 56 * v13 + 8;
}
