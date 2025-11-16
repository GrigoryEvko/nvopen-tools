// Function: sub_1DFBD00
// Address: 0x1dfbd00
//
unsigned __int64 __fastcall sub_1DFBD00(__int64 a1, __int64 a2, __int64 *a3, int *a4)
{
  char v7; // al
  __int64 *v8; // r14
  unsigned __int64 result; // rax
  __int64 v10; // r9
  __int64 v11; // r8
  int v12; // edx
  unsigned int v13; // esi
  int v14; // eax
  int v15; // eax
  __int64 v16; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  unsigned __int64 v19; // r10
  __int64 v20; // rax
  unsigned __int64 v21; // r10
  __int64 v22; // rdx
  unsigned __int64 v23; // r10
  __int64 v24; // [rsp+0h] [rbp-60h]
  __int64 v25; // [rsp+0h] [rbp-60h]
  __int64 v26; // [rsp+8h] [rbp-58h]
  __int64 v27; // [rsp+8h] [rbp-58h]
  unsigned int v28; // [rsp+10h] [rbp-50h]
  unsigned __int64 v29; // [rsp+10h] [rbp-50h]
  unsigned __int64 v30; // [rsp+18h] [rbp-48h]
  __int64 v31; // [rsp+18h] [rbp-48h]
  __int64 *v32[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = sub_1DF9540(a1, a3, v32);
  v8 = v32[0];
  if ( !v7 )
  {
    v13 = *(_DWORD *)(a1 + 24);
    v14 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)a1;
    v15 = v14 + 1;
    if ( 4 * v15 >= 3 * v13 )
    {
      v13 *= 2;
    }
    else if ( v13 - *(_DWORD *)(a1 + 20) - v15 > v13 >> 3 )
    {
      goto LABEL_7;
    }
    sub_1DFBB90(a1, v13);
    sub_1DF9540(a1, a3, v32);
    v8 = v32[0];
    v15 = *(_DWORD *)(a1 + 16) + 1;
LABEL_7:
    *(_DWORD *)(a1 + 16) = v15;
    if ( *v8 )
      --*(_DWORD *)(a1 + 20);
    v16 = *a3;
    v8[1] = 0;
    v10 = 0;
    *v8 = v16;
    result = *(_QWORD *)(a1 + 40);
    v11 = *(_QWORD *)(a2 + 16);
    if ( result )
      goto LABEL_3;
    goto LABEL_10;
  }
  result = *(_QWORD *)(a1 + 40);
  v10 = v32[0][1];
  v11 = *(_QWORD *)(a2 + 16);
  if ( result )
  {
LABEL_3:
    *(_QWORD *)(a1 + 40) = *(_QWORD *)result;
    goto LABEL_4;
  }
LABEL_10:
  v17 = *(_QWORD *)(a1 + 48);
  v18 = *(_QWORD *)(a1 + 56);
  *(_QWORD *)(a1 + 128) += 32LL;
  if ( ((v17 + 7) & 0xFFFFFFFFFFFFFFF8LL) - v17 + 32 <= v18 - v17 )
  {
    result = (v17 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    *(_QWORD *)(a1 + 48) = result + 32;
  }
  else
  {
    v24 = v10;
    v19 = 0x40000000000LL;
    v26 = v11;
    v28 = *(_DWORD *)(a1 + 72);
    if ( v28 >> 7 < 0x1E )
      v19 = 4096LL << (v28 >> 7);
    v30 = v19;
    v20 = malloc(v19);
    v21 = v30;
    v22 = v28;
    v11 = v26;
    v10 = v24;
    if ( !v20 )
    {
      sub_16BD1C0("Allocation failed", 1u);
      v22 = *(unsigned int *)(a1 + 72);
      v10 = v24;
      v20 = 0;
      v21 = v30;
      v11 = v26;
    }
    if ( (unsigned int)v22 >= *(_DWORD *)(a1 + 76) )
    {
      v25 = v10;
      v27 = v20;
      v29 = v21;
      v31 = v11;
      sub_16CD150(a1 + 64, (const void *)(a1 + 80), 0, 8, v11, v10);
      v22 = *(unsigned int *)(a1 + 72);
      v10 = v25;
      v20 = v27;
      v21 = v29;
      v11 = v31;
    }
    v23 = v20 + v21;
    *(_QWORD *)(*(_QWORD *)(a1 + 64) + 8 * v22) = v20;
    result = (v20 + 7) & 0xFFFFFFFFFFFFFFF8LL;
    ++*(_DWORD *)(a1 + 72);
    *(_QWORD *)(a1 + 56) = v23;
    *(_QWORD *)(a1 + 48) = result + 32;
  }
  if ( !result )
  {
    MEMORY[0] = v11;
    BUG();
  }
LABEL_4:
  *(_QWORD *)(result + 16) = *a3;
  v12 = *a4;
  *(_QWORD *)result = v11;
  *(_DWORD *)(result + 24) = v12;
  *(_QWORD *)(result + 8) = v10;
  v8[1] = result;
  *(_QWORD *)(a2 + 16) = result;
  return result;
}
