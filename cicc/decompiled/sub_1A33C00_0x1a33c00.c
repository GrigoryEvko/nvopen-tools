// Function: sub_1A33C00
// Address: 0x1a33c00
//
__int64 __fastcall sub_1A33C00(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  __int64 v8; // rcx
  __int64 v9; // rdi
  char v10; // al
  __int64 v11; // r8
  __int64 *v12; // rdx
  __int64 v13; // r15
  __int64 v15; // r15
  int v16; // r10d
  unsigned int v17; // r9d
  unsigned int v18; // edx
  __int64 *v19; // rsi
  int v20; // ecx
  unsigned int v21; // eax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rdx
  __int64 v26; // rax
  __int64 v27; // rcx
  __int64 v28; // r11
  int v29; // r15d
  int v30; // esi
  __int64 *v31; // [rsp+18h] [rbp-38h] BYREF

  v8 = *(_QWORD *)(a2 + 16);
  v9 = *(_QWORD *)a2;
  v10 = *(_BYTE *)(a2 + 8) & 1;
  if ( v10 )
  {
    v11 = *a3;
    v12 = (__int64 *)(a2 + 16);
    if ( *a3 == v8 )
    {
      v8 = a2 + 16;
      v13 = 32;
LABEL_4:
      *(_QWORD *)a1 = a2;
      *(_QWORD *)(a1 + 8) = v9;
      *(_QWORD *)(a1 + 16) = v12;
      *(_QWORD *)(a1 + 24) = v8 + v13;
      *(_BYTE *)(a1 + 32) = 0;
      return a1;
    }
    v28 = a2 + 16;
    v17 = 0;
    v16 = 0;
    goto LABEL_19;
  }
  v15 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v15 )
  {
    v11 = *a3;
    v16 = v15 - 1;
    v17 = (v15 - 1) & (((unsigned int)*a3 >> 4) ^ ((unsigned int)*a3 >> 9));
    v12 = (__int64 *)(v8 + 32LL * v17);
    if ( v11 == *v12 )
    {
LABEL_8:
      v13 = 32 * v15;
      goto LABEL_4;
    }
    v28 = *(_QWORD *)(a2 + 16);
    v8 = *v12;
LABEL_19:
    v29 = 1;
    v19 = 0;
    while ( 1 )
    {
      if ( v8 == -8 )
      {
        if ( !v19 )
          v19 = v12;
        v18 = *(_DWORD *)(a2 + 8);
        *(_QWORD *)a2 = v9 + 1;
        v20 = (v18 >> 1) + 1;
        if ( v10 )
        {
          v21 = 3;
          LODWORD(v15) = 1;
          goto LABEL_11;
        }
        LODWORD(v15) = *(_DWORD *)(a2 + 24);
        goto LABEL_10;
      }
      if ( !v19 && v8 == -16 )
        v19 = v12;
      v17 = v16 & (v29 + v17);
      v12 = (__int64 *)(v28 + 32LL * v17);
      v8 = *v12;
      if ( *v12 == v11 )
        break;
      ++v29;
    }
    if ( !v10 )
    {
      v15 = *(unsigned int *)(a2 + 24);
      v8 = v28;
      goto LABEL_8;
    }
    v8 = v28;
    v13 = 32;
    goto LABEL_4;
  }
  v18 = *(_DWORD *)(a2 + 8);
  *(_QWORD *)a2 = v9 + 1;
  v19 = 0;
  v20 = (v18 >> 1) + 1;
LABEL_10:
  v21 = 3 * v15;
LABEL_11:
  if ( 4 * v20 >= v21 )
  {
    v30 = 2 * v15;
LABEL_27:
    sub_1A33A60(a2, v30);
    sub_1A27450(a2, a3, &v31);
    v19 = v31;
    v18 = *(_DWORD *)(a2 + 8);
    goto LABEL_13;
  }
  if ( (int)v15 - *(_DWORD *)(a2 + 12) - v20 <= (unsigned int)v15 >> 3 )
  {
    v30 = v15;
    goto LABEL_27;
  }
LABEL_13:
  *(_DWORD *)(a2 + 8) = (2 * (v18 >> 1) + 2) | v18 & 1;
  if ( *v19 != -8 )
    --*(_DWORD *)(a2 + 12);
  *v19 = *a3;
  v22 = *a4;
  *a4 = 0;
  v19[1] = v22;
  v23 = a4[1];
  a4[1] = 0;
  v19[2] = v23;
  v24 = a4[2];
  a4[2] = 0;
  v19[3] = v24;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v25 = a2 + 16;
    v26 = 32;
  }
  else
  {
    v25 = *(_QWORD *)(a2 + 16);
    v26 = 32LL * *(unsigned int *)(a2 + 24);
  }
  v27 = *(_QWORD *)a2;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v19;
  *(_QWORD *)(a1 + 8) = v27;
  *(_QWORD *)(a1 + 24) = v25 + v26;
  *(_BYTE *)(a1 + 32) = 1;
  return a1;
}
