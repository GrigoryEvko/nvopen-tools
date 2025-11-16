// Function: sub_1913110
// Address: 0x1913110
//
__int64 __fastcall sub_1913110(__int64 a1, __int64 a2, int a3, unsigned int a4, __int64 a5, __int64 a6)
{
  const void *v8; // rbx
  _QWORD **v9; // rax
  __int64 *v10; // rax
  __int64 v11; // rax
  __int64 v12; // r8
  int v13; // r8d
  int v14; // r9d
  __int64 v15; // rax
  int v16; // r8d
  int v17; // r9d
  int v18; // r13d
  __int64 v19; // rax
  unsigned int *v20; // rax
  unsigned int v21; // edx
  unsigned int v22; // ecx
  int v25; // [rsp+8h] [rbp-48h]
  _QWORD *v26; // [rsp+10h] [rbp-40h]
  __int64 v27; // [rsp+10h] [rbp-40h]

  v8 = (const void *)(a1 + 40);
  *(_DWORD *)a1 = -3;
  *(_BYTE *)(a1 + 16) = 0;
  *(_QWORD *)(a1 + 24) = a1 + 40;
  *(_QWORD *)(a1 + 32) = 0x400000000LL;
  v9 = *(_QWORD ***)a5;
  if ( *(_BYTE *)(*(_QWORD *)a5 + 8LL) == 16 )
  {
    v26 = v9[4];
    v10 = (__int64 *)sub_1643320(*v9);
    v11 = (__int64)sub_16463B0(v10, (unsigned int)v26);
    v12 = a5;
  }
  else
  {
    v11 = sub_1643320(*v9);
    v12 = a5;
  }
  *(_QWORD *)(a1 + 8) = v11;
  v27 = a1 + 24;
  v13 = sub_1911FD0(a2, v12);
  v15 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)v15 >= *(_DWORD *)(a1 + 36) )
  {
    v25 = v13;
    sub_16CD150(v27, v8, 0, 4, v13, v14);
    v15 = *(unsigned int *)(a1 + 32);
    v13 = v25;
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 24) + 4 * v15) = v13;
  ++*(_DWORD *)(a1 + 32);
  v18 = sub_1911FD0(a2, a6);
  v19 = *(unsigned int *)(a1 + 32);
  if ( (unsigned int)v19 >= *(_DWORD *)(a1 + 36) )
  {
    sub_16CD150(v27, v8, 0, 4, v16, v17);
    v19 = *(unsigned int *)(a1 + 32);
  }
  *(_DWORD *)(*(_QWORD *)(a1 + 24) + 4 * v19) = v18;
  v20 = *(unsigned int **)(a1 + 24);
  ++*(_DWORD *)(a1 + 32);
  v21 = *v20;
  v22 = v20[1];
  if ( *v20 > v22 )
  {
    *v20 = v22;
    v20[1] = v21;
    a4 = sub_15FF5D0(a4);
  }
  *(_BYTE *)(a1 + 16) = 1;
  *(_DWORD *)a1 = (a3 << 8) | a4;
  return a1;
}
