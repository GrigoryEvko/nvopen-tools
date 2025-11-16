// Function: sub_2AE5BD0
// Address: 0x2ae5bd0
//
__int64 __fastcall sub_2AE5BD0(__int64 a1, int *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v8; // eax
  char v9; // dl
  int v10; // edx
  __int64 v11; // r8
  int v12; // esi
  unsigned int v13; // ecx
  unsigned int *v14; // rdi
  __int64 v15; // rax
  unsigned int v17; // esi
  unsigned int v18; // ecx
  int v19; // edi
  __int64 v20; // r8
  int *v21; // r13
  __int64 v22; // rax
  __int64 v23; // r12
  int v24; // r11d
  unsigned int *v25; // r10
  int *v26; // [rsp+0h] [rbp-30h] BYREF
  int v27; // [rsp+8h] [rbp-28h] BYREF
  int v28; // [rsp+Ch] [rbp-24h]

  v8 = *a2;
  v9 = *(_BYTE *)(a1 + 8);
  v28 = 0;
  v27 = v8;
  v10 = v9 & 1;
  if ( v10 )
  {
    v11 = a1 + 16;
    v12 = 3;
  }
  else
  {
    v17 = *(_DWORD *)(a1 + 24);
    v11 = *(_QWORD *)(a1 + 16);
    if ( !v17 )
    {
      v18 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v26 = 0;
      v19 = (v18 >> 1) + 1;
LABEL_9:
      v20 = 3 * v17;
      goto LABEL_10;
    }
    v12 = v17 - 1;
  }
  v13 = v12 & (37 * v8);
  v14 = (unsigned int *)(v11 + 8LL * v13);
  a6 = *v14;
  if ( v8 == (_DWORD)a6 )
  {
LABEL_4:
    v15 = v14[1];
    return *(_QWORD *)(a1 + 48) + 8 * v15 + 4;
  }
  v24 = 1;
  v25 = 0;
  while ( (_DWORD)a6 != -1 )
  {
    if ( !v25 && (_DWORD)a6 == -2 )
      v25 = v14;
    v13 = v12 & (v24 + v13);
    v14 = (unsigned int *)(v11 + 8LL * v13);
    a6 = *v14;
    if ( v8 == (_DWORD)a6 )
      goto LABEL_4;
    ++v24;
  }
  v18 = *(_DWORD *)(a1 + 8);
  if ( !v25 )
    v25 = v14;
  ++*(_QWORD *)a1;
  v26 = (int *)v25;
  v19 = (v18 >> 1) + 1;
  if ( !(_BYTE)v10 )
  {
    v17 = *(_DWORD *)(a1 + 24);
    goto LABEL_9;
  }
  v20 = 12;
  v17 = 4;
LABEL_10:
  if ( 4 * v19 >= (unsigned int)v20 )
  {
    v17 *= 2;
    goto LABEL_24;
  }
  if ( v17 - *(_DWORD *)(a1 + 12) - v19 <= v17 >> 3 )
  {
LABEL_24:
    sub_29758F0(a1, v17);
    sub_2AC3C60(a1, &v27, &v26);
    v8 = v27;
    v18 = *(_DWORD *)(a1 + 8);
  }
  v21 = v26;
  *(_DWORD *)(a1 + 8) = (2 * (v18 >> 1) + 2) | v18 & 1;
  if ( *v21 != -1 )
    --*(_DWORD *)(a1 + 12);
  *v21 = v8;
  v21[1] = v28;
  v22 = *(unsigned int *)(a1 + 56);
  v23 = (unsigned int)*a2;
  if ( v22 + 1 > (unsigned __int64)*(unsigned int *)(a1 + 60) )
  {
    sub_C8D5F0(a1 + 48, (const void *)(a1 + 64), v22 + 1, 8u, v20, a6);
    v22 = *(unsigned int *)(a1 + 56);
  }
  *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8 * v22) = v23;
  v15 = *(unsigned int *)(a1 + 56);
  *(_DWORD *)(a1 + 56) = v15 + 1;
  v21[1] = v15;
  return *(_QWORD *)(a1 + 48) + 8 * v15 + 4;
}
