// Function: sub_2B4D5E0
// Address: 0x2b4d5e0
//
__int64 __fastcall sub_2B4D5E0(__int64 a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v8; // rsi
  __int64 v9; // r9
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 *v12; // r14
  int v13; // r15d
  unsigned int v14; // edx
  __int64 *v15; // rdi
  __int64 v16; // r10
  __int64 v17; // r9
  int v18; // ecx
  __int64 v19; // rdx
  __int64 v20; // rdx
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // rdx
  __int64 v24; // rdx
  __int64 v25; // rcx
  int v26; // edx
  int v27; // esi
  unsigned int v28; // edx
  int v29; // r11d
  __int64 *v30; // r10
  int v31; // edx
  int v32; // esi
  int v33; // r11d
  unsigned int v34; // edx
  __int64 v35; // [rsp+8h] [rbp-38h]
  __int64 v36; // [rsp+8h] [rbp-38h]
  __int64 v37; // [rsp+8h] [rbp-38h]

  result = a1;
  v8 = *(unsigned int *)(a2 + 24);
  v9 = *(_QWORD *)a2;
  if ( !(_DWORD)v8 )
  {
    *(_QWORD *)a2 = v9 + 1;
    goto LABEL_20;
  }
  v10 = *a3;
  v11 = *(_QWORD *)(a2 + 8);
  v12 = 0;
  v13 = 1;
  v14 = (v8 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
  v15 = (__int64 *)(v11 + 40LL * v14);
  v16 = *v15;
  if ( v10 == *v15 )
  {
LABEL_3:
    *(_QWORD *)result = a2;
    *(_QWORD *)(result + 8) = v9;
    *(_QWORD *)(result + 16) = v15;
    *(_QWORD *)(result + 24) = v11 + 40 * v8;
    *(_BYTE *)(result + 32) = 0;
    return result;
  }
  while ( v16 != -4096 )
  {
    if ( v16 == -8192 && !v12 )
      v12 = v15;
    v14 = (v8 - 1) & (v13 + v14);
    v15 = (__int64 *)(v11 + 40LL * v14);
    v16 = *v15;
    if ( v10 == *v15 )
      goto LABEL_3;
    ++v13;
  }
  if ( !v12 )
    v12 = v15;
  v17 = v9 + 1;
  v18 = *(_DWORD *)(a2 + 16) + 1;
  *(_QWORD *)a2 = v17;
  if ( 4 * v18 >= (unsigned int)(3 * v8) )
  {
LABEL_20:
    v36 = result;
    sub_2B4D200(a2, 2 * v8);
    v26 = *(_DWORD *)(a2 + 24);
    if ( v26 )
    {
      v27 = v26 - 1;
      v11 = *(_QWORD *)(a2 + 8);
      v28 = (v26 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
      v12 = (__int64 *)(v11 + 40LL * v28);
      v18 = *(_DWORD *)(a2 + 16) + 1;
      result = v36;
      v17 = *v12;
      if ( *v12 == *a3 )
        goto LABEL_14;
      v29 = 1;
      v30 = 0;
      while ( v17 != -4096 )
      {
        if ( !v30 && v17 == -8192 )
          v30 = v12;
        v28 = v27 & (v29 + v28);
        v12 = (__int64 *)(v11 + 40LL * v28);
        v17 = *v12;
        if ( *a3 == *v12 )
          goto LABEL_14;
        ++v29;
      }
LABEL_24:
      if ( v30 )
        v12 = v30;
      goto LABEL_14;
    }
LABEL_40:
    ++*(_DWORD *)(a2 + 16);
    BUG();
  }
  if ( (int)v8 - *(_DWORD *)(a2 + 20) - v18 <= (unsigned int)v8 >> 3 )
  {
    v37 = result;
    sub_2B4D200(a2, v8);
    v31 = *(_DWORD *)(a2 + 24);
    if ( v31 )
    {
      v32 = v31 - 1;
      v11 = *(_QWORD *)(a2 + 8);
      v30 = 0;
      v33 = 1;
      v34 = (v31 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
      v12 = (__int64 *)(v11 + 40LL * v34);
      v18 = *(_DWORD *)(a2 + 16) + 1;
      result = v37;
      v17 = *v12;
      if ( *a3 == *v12 )
        goto LABEL_14;
      while ( v17 != -4096 )
      {
        if ( v17 == -8192 && !v30 )
          v30 = v12;
        v34 = v32 & (v33 + v34);
        v12 = (__int64 *)(v11 + 40LL * v34);
        v17 = *v12;
        if ( *a3 == *v12 )
          goto LABEL_14;
        ++v33;
      }
      goto LABEL_24;
    }
    goto LABEL_40;
  }
LABEL_14:
  *(_DWORD *)(a2 + 16) = v18;
  if ( *v12 != -4096 )
    --*(_DWORD *)(a2 + 20);
  v19 = *a3;
  v12[2] = 0x400000000LL;
  *v12 = v19;
  v12[1] = (__int64)(v12 + 3);
  v20 = *(unsigned int *)(a4 + 8);
  if ( (_DWORD)v20 )
  {
    v35 = result;
    sub_2B0D430((__int64)(v12 + 1), a4, v20, 0x400000000LL, v11, v17);
    result = v35;
  }
  v21 = *(unsigned int *)(a2 + 24);
  *(_QWORD *)result = a2;
  *(_QWORD *)(result + 16) = v12;
  v22 = 5 * v21;
  v23 = *(_QWORD *)(a2 + 8);
  *(_BYTE *)(result + 32) = 1;
  v24 = v23 + 8 * v22;
  v25 = *(_QWORD *)a2;
  *(_QWORD *)(result + 24) = v24;
  *(_QWORD *)(result + 8) = v25;
  return result;
}
