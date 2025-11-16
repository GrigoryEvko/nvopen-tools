// Function: sub_FB0400
// Address: 0xfb0400
//
__int64 __fastcall sub_FB0400(__int64 a1, __int64 a2, __int64 *a3, __int64 *a4)
{
  __int64 v8; // rdi
  char v9; // cl
  __int64 v10; // r8
  __int64 v11; // rsi
  __int64 v12; // r11
  int v13; // r10d
  unsigned int v14; // edx
  __int64 *v15; // rax
  __int64 v16; // r9
  __int64 v18; // r11
  unsigned int v19; // eax
  int v20; // edx
  unsigned int v21; // esi
  __int64 *v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // rsi
  int v26; // r15d
  __int64 *v27; // r11
  __int64 v28; // rdi
  int v29; // esi
  unsigned int v30; // eax
  __int64 v31; // r8
  int v32; // eax
  int v33; // r10d
  __int64 *v34; // r9
  __int64 *v35; // [rsp+18h] [rbp-38h] BYREF

  v8 = *(_QWORD *)a2;
  v9 = *(_BYTE *)(a2 + 8) & 1;
  if ( v9 )
  {
    v10 = a2 + 16;
    v11 = *a3;
    v12 = 64;
    v13 = 3;
    v14 = ((unsigned __int8)((unsigned int)*a3 >> 4) ^ (unsigned __int8)((unsigned int)*a3 >> 9)) & 3;
    v15 = (__int64 *)(v10 + 16LL * v14);
    v16 = *v15;
    if ( v11 == *v15 )
    {
LABEL_3:
      *(_QWORD *)a1 = a2;
      *(_QWORD *)(a1 + 8) = v8;
      *(_QWORD *)(a1 + 16) = v15;
      *(_QWORD *)(a1 + 24) = v10 + v12;
      *(_BYTE *)(a1 + 32) = 0;
      return a1;
    }
    goto LABEL_19;
  }
  v18 = *(unsigned int *)(a2 + 24);
  if ( (_DWORD)v18 )
  {
    v10 = *(_QWORD *)(a2 + 16);
    v11 = *a3;
    v13 = v18 - 1;
    v14 = (v18 - 1) & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
    v15 = (__int64 *)(v10 + 16LL * v14);
    v16 = *v15;
    if ( v11 == *v15 )
    {
LABEL_7:
      v12 = 16 * v18;
      goto LABEL_3;
    }
LABEL_19:
    v26 = 1;
    v27 = 0;
    while ( 1 )
    {
      if ( v16 == -4096 )
      {
        if ( !v27 )
          v27 = v15;
        v19 = *(_DWORD *)(a2 + 8);
        *(_QWORD *)a2 = v8 + 1;
        v35 = v27;
        v20 = (v19 >> 1) + 1;
        if ( v9 )
        {
          v21 = 12;
          LODWORD(v18) = 4;
          goto LABEL_10;
        }
        LODWORD(v18) = *(_DWORD *)(a2 + 24);
        goto LABEL_9;
      }
      if ( v16 == -8192 && !v27 )
        v27 = v15;
      v14 = v13 & (v26 + v14);
      v15 = (__int64 *)(v10 + 16LL * v14);
      v16 = *v15;
      if ( v11 == *v15 )
        break;
      ++v26;
    }
    if ( !v9 )
    {
      v18 = *(unsigned int *)(a2 + 24);
      goto LABEL_7;
    }
    v12 = 64;
    goto LABEL_3;
  }
  v19 = *(_DWORD *)(a2 + 8);
  v35 = 0;
  *(_QWORD *)a2 = v8 + 1;
  v20 = (v19 >> 1) + 1;
LABEL_9:
  v21 = 3 * v18;
LABEL_10:
  if ( 4 * v20 < v21 )
  {
    if ( (int)v18 - *(_DWORD *)(a2 + 12) - v20 <= (unsigned int)v18 >> 3 )
    {
      sub_FAFFE0(a2, v18);
      sub_F9ED60(a2, a3, &v35);
      v22 = v35;
      v19 = *(_DWORD *)(a2 + 8);
    }
    else
    {
      v22 = v35;
    }
    goto LABEL_13;
  }
  sub_FAFFE0(a2, 2 * v18);
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v28 = a2 + 16;
    v29 = 3;
    goto LABEL_27;
  }
  v32 = *(_DWORD *)(a2 + 24);
  v28 = *(_QWORD *)(a2 + 16);
  v29 = v32 - 1;
  if ( v32 )
  {
LABEL_27:
    v30 = v29 & (((unsigned int)*a3 >> 9) ^ ((unsigned int)*a3 >> 4));
    v22 = (__int64 *)(v28 + 16LL * v30);
    v31 = *v22;
    if ( *v22 == *a3 )
    {
LABEL_28:
      v35 = v22;
      v19 = *(_DWORD *)(a2 + 8);
    }
    else
    {
      v33 = 1;
      v34 = 0;
      while ( v31 != -4096 )
      {
        if ( !v34 && v31 == -8192 )
          v34 = v22;
        v30 = v29 & (v33 + v30);
        v22 = (__int64 *)(v28 + 16LL * v30);
        v31 = *v22;
        if ( *a3 == *v22 )
          goto LABEL_28;
        ++v33;
      }
      v19 = *(_DWORD *)(a2 + 8);
      if ( !v34 )
        v34 = v22;
      v35 = v34;
      v22 = v34;
    }
    goto LABEL_13;
  }
  v35 = 0;
  v19 = *(_DWORD *)(a2 + 8);
  v22 = 0;
LABEL_13:
  *(_DWORD *)(a2 + 8) = (2 * (v19 >> 1) + 2) | v19 & 1;
  if ( *v22 != -4096 )
    --*(_DWORD *)(a2 + 12);
  *v22 = *a3;
  v22[1] = *a4;
  if ( (*(_BYTE *)(a2 + 8) & 1) != 0 )
  {
    v23 = a2 + 16;
    v24 = 64;
  }
  else
  {
    v23 = *(_QWORD *)(a2 + 16);
    v24 = 16LL * *(unsigned int *)(a2 + 24);
  }
  v25 = *(_QWORD *)a2;
  *(_QWORD *)a1 = a2;
  *(_QWORD *)(a1 + 16) = v22;
  *(_QWORD *)(a1 + 8) = v25;
  *(_QWORD *)(a1 + 24) = v24 + v23;
  *(_BYTE *)(a1 + 32) = 1;
  return a1;
}
