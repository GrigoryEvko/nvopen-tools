// Function: sub_256F330
// Address: 0x256f330
//
__int64 __fastcall sub_256F330(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v8; // rdx
  char v9; // di
  int v10; // edi
  __int64 v11; // r8
  int v12; // esi
  unsigned int v13; // ecx
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned int v17; // esi
  unsigned int v18; // eax
  int v19; // ecx
  __int64 v20; // r8
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // r12
  __int64 *v24; // rax
  int v25; // r11d
  __int64 v26; // r10
  __int64 v27; // [rsp+8h] [rbp-38h] BYREF
  __int64 v28; // [rsp+10h] [rbp-30h] BYREF
  int v29; // [rsp+18h] [rbp-28h]

  v8 = *a2;
  v9 = *(_BYTE *)(a1 + 8);
  v29 = 0;
  v28 = v8;
  v10 = v9 & 1;
  if ( v10 )
  {
    v11 = a1 + 16;
    v12 = 31;
  }
  else
  {
    v17 = *(_DWORD *)(a1 + 24);
    v11 = *(_QWORD *)(a1 + 16);
    if ( !v17 )
    {
      v18 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v27 = 0;
      v19 = (v18 >> 1) + 1;
LABEL_9:
      v20 = 3 * v17;
      goto LABEL_10;
    }
    v12 = v17 - 1;
  }
  v13 = v12 & (((unsigned int)v8 >> 9) ^ ((unsigned int)v8 >> 4));
  v14 = v11 + 16LL * v13;
  a6 = *(_QWORD *)v14;
  if ( v8 == *(_QWORD *)v14 )
  {
LABEL_4:
    v15 = *(unsigned int *)(v14 + 8);
    return *(_QWORD *)(a1 + 528) + 16 * v15 + 8;
  }
  v25 = 1;
  v26 = 0;
  while ( a6 != -4096 )
  {
    if ( !v26 && a6 == -8192 )
      v26 = v14;
    v13 = v12 & (v25 + v13);
    v14 = v11 + 16LL * v13;
    a6 = *(_QWORD *)v14;
    if ( v8 == *(_QWORD *)v14 )
      goto LABEL_4;
    ++v25;
  }
  if ( !v26 )
    v26 = v14;
  v18 = *(_DWORD *)(a1 + 8);
  ++*(_QWORD *)a1;
  v27 = v26;
  v19 = (v18 >> 1) + 1;
  if ( !(_BYTE)v10 )
  {
    v17 = *(_DWORD *)(a1 + 24);
    goto LABEL_9;
  }
  v20 = 96;
  v17 = 32;
LABEL_10:
  if ( 4 * v19 >= (unsigned int)v20 )
  {
    v17 *= 2;
    goto LABEL_24;
  }
  if ( v17 - *(_DWORD *)(a1 + 12) - v19 <= v17 >> 3 )
  {
LABEL_24:
    sub_256EF10(a1, v17);
    sub_2566B70(a1, &v28, &v27);
    v8 = v28;
    v18 = *(_DWORD *)(a1 + 8);
  }
  v21 = v27;
  *(_DWORD *)(a1 + 8) = (2 * (v18 >> 1) + 2) | v18 & 1;
  if ( *(_QWORD *)v21 != -4096 )
    --*(_DWORD *)(a1 + 12);
  *(_QWORD *)v21 = v8;
  *(_DWORD *)(v21 + 8) = v29;
  v22 = *(unsigned int *)(a1 + 536);
  v23 = *a2;
  if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 540) )
  {
    sub_C8D5F0(a1 + 528, (const void *)(a1 + 544), v22 + 1, 0x10u, v20, a6);
    v22 = *(unsigned int *)(a1 + 536);
  }
  v24 = (__int64 *)(*(_QWORD *)(a1 + 528) + 16 * v22);
  *v24 = v23;
  v24[1] = 0;
  v15 = *(unsigned int *)(a1 + 536);
  *(_DWORD *)(a1 + 536) = v15 + 1;
  *(_DWORD *)(v21 + 8) = v15;
  return *(_QWORD *)(a1 + 528) + 16 * v15 + 8;
}
