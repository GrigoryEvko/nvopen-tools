// Function: sub_2D7B240
// Address: 0x2d7b240
//
__int64 __fastcall sub_2D7B240(__int64 a1, __int64 *a2)
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
  char **v17; // r14
  __int64 v18; // rdx
  unsigned __int64 v19; // rcx
  __int64 v20; // rdx
  unsigned __int64 v21; // rsi
  __int64 v22; // rcx
  __int64 v23; // rdx
  char **v24; // rdi
  _BYTE *v25; // rdi
  unsigned __int64 v26; // r15
  __int64 v27; // rdi
  __int64 v28; // [rsp+0h] [rbp-170h] BYREF
  int v29; // [rsp+8h] [rbp-168h]
  char *v30; // [rsp+10h] [rbp-160h]
  __int64 v31; // [rsp+18h] [rbp-158h]
  char v32; // [rsp+20h] [rbp-150h] BYREF
  __int64 v33; // [rsp+A0h] [rbp-D0h] BYREF
  _BYTE *v34; // [rsp+A8h] [rbp-C8h]
  __int64 v35; // [rsp+B0h] [rbp-C0h]
  _BYTE v36[184]; // [rsp+B8h] [rbp-B8h] BYREF

  v4 = *a2;
  v5 = *(_DWORD *)(a1 + 24);
  v29 = 0;
  v28 = v4;
  if ( !v5 )
  {
    ++*(_QWORD *)a1;
    v33 = 0;
LABEL_27:
    v17 = (char **)&v33;
    sub_D39D40(a1, 2 * v5);
LABEL_28:
    sub_22B1A50(a1, &v28, &v33);
    v4 = v28;
    v9 = v33;
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
    return *(_QWORD *)(a1 + 32) + 152 * v13 + 8;
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
  v33 = v9;
  if ( 4 * (v15 + 1) >= 3 * v5 )
    goto LABEL_27;
  v17 = (char **)&v33;
  if ( v5 - *(_DWORD *)(a1 + 20) - v16 <= v5 >> 3 )
  {
    sub_D39D40(a1, v5);
    goto LABEL_28;
  }
LABEL_15:
  *(_DWORD *)(a1 + 16) = v16;
  if ( *(_QWORD *)v9 != -4096 )
    --*(_DWORD *)(a1 + 20);
  *(_QWORD *)v9 = v4;
  *(_DWORD *)(v9 + 8) = v29;
  v18 = *a2;
  v19 = *(unsigned int *)(a1 + 44);
  v30 = &v32;
  v33 = v18;
  v20 = *(unsigned int *)(a1 + 40);
  v31 = 0x1000000000LL;
  v21 = v20 + 1;
  v35 = 0x1000000000LL;
  v13 = v20;
  v34 = v36;
  if ( v20 + 1 > v19 )
  {
    v26 = *(_QWORD *)(a1 + 32);
    v27 = a1 + 32;
    if ( v26 > (unsigned __int64)&v33 || (unsigned __int64)&v33 >= v26 + 152 * v20 )
    {
      sub_2D71A30(v27, v21, v20, v19, v6, v12);
      v20 = *(unsigned int *)(a1 + 40);
      v22 = *(_QWORD *)(a1 + 32);
      v13 = v20;
    }
    else
    {
      sub_2D71A30(v27, v21, v20, v19, v6, v12);
      v22 = *(_QWORD *)(a1 + 32);
      v20 = *(unsigned int *)(a1 + 40);
      v17 = (char **)((char *)&v33 + v22 - v26);
      v13 = v20;
    }
  }
  else
  {
    v22 = *(_QWORD *)(a1 + 32);
  }
  v23 = 19 * v20;
  v24 = (char **)(v22 + 8 * v23);
  if ( v24 )
  {
    *v24 = *v17;
    v24[1] = (char *)(v24 + 3);
    v24[2] = (char *)0x1000000000LL;
    if ( *((_DWORD *)v17 + 4) )
      sub_2D56D30((__int64)(v24 + 1), v17 + 1, v23, v22, v6, v12);
    v13 = *(unsigned int *)(a1 + 40);
  }
  v25 = v34;
  *(_DWORD *)(a1 + 40) = v13 + 1;
  if ( v25 != v36 )
  {
    _libc_free((unsigned __int64)v25);
    v13 = (unsigned int)(*(_DWORD *)(a1 + 40) - 1);
  }
  *(_DWORD *)(v9 + 8) = v13;
  return *(_QWORD *)(a1 + 32) + 152 * v13 + 8;
}
