// Function: sub_1A25930
// Address: 0x1a25930
//
__int64 __fastcall sub_1A25930(__int64 *a1, __int64 a2)
{
  __int64 v4; // r13
  _BOOL4 v5; // r8d
  __int64 result; // rax
  __int64 v7; // r14
  char v8; // al
  __int64 v9; // rsi
  int v10; // ecx
  int v11; // r8d
  unsigned int v12; // edx
  __int64 *v13; // r15
  __int64 v14; // rdi
  int v15; // edi
  __int64 v16; // r8
  unsigned int v17; // eax
  __int64 *v18; // rcx
  __int64 v19; // r9
  const void *v20; // rsi
  size_t v21; // rdx
  const void *v22; // rdi
  unsigned int v23; // esi
  __int64 *v24; // rdx
  unsigned int v25; // eax
  __int64 v26; // rax
  __int64 v27; // rdx
  __int64 v28; // r15
  int v29; // r11d
  int v30; // esi
  __int64 *v31; // r10
  __int64 v32; // [rsp+0h] [rbp-40h] BYREF
  __int64 *v33; // [rsp+8h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a2 - 48);
  v5 = sub_1A25850(*a1, v4);
  result = 1;
  if ( !v5 )
  {
    v7 = a1[1];
    v8 = *(_BYTE *)(v7 + 8) & 1;
    if ( v8 )
    {
      v9 = v7 + 16;
      v10 = 7;
    }
    else
    {
      v27 = *(unsigned int *)(v7 + 24);
      v9 = *(_QWORD *)(v7 + 16);
      if ( !(_DWORD)v27 )
        goto LABEL_27;
      v10 = v27 - 1;
    }
    v11 = 1;
    v12 = v10 & (((unsigned int)v4 >> 9) ^ ((unsigned int)v4 >> 4));
    v13 = (__int64 *)(v9 + 40LL * v12);
    v14 = *v13;
    if ( v4 == *v13 )
      goto LABEL_5;
    while ( v14 != -8 )
    {
      v12 = v10 & (v11 + v12);
      v13 = (__int64 *)(v9 + 40LL * v12);
      v14 = *v13;
      if ( v4 == *v13 )
        goto LABEL_5;
      ++v11;
    }
    if ( v8 )
    {
      v28 = 320;
      goto LABEL_28;
    }
    v27 = *(unsigned int *)(v7 + 24);
LABEL_27:
    v28 = 40 * v27;
LABEL_28:
    v13 = (__int64 *)(v9 + v28);
LABEL_5:
    if ( v8 )
    {
      if ( v13 == (__int64 *)(v9 + 320) )
        return 0;
      v32 = a2;
      v15 = 7;
      LODWORD(v16) = 8;
    }
    else
    {
      v16 = *(unsigned int *)(v7 + 24);
      if ( v13 == (__int64 *)(v9 + 40 * v16) )
        return 0;
      v32 = a2;
      v15 = v16 - 1;
      if ( !(_DWORD)v16 )
      {
        ++*(_QWORD *)v7;
        goto LABEL_15;
      }
    }
    v17 = v15 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
    v18 = (__int64 *)(v9 + 40LL * v17);
    v19 = *v18;
    if ( a2 == *v18 )
    {
      v20 = (const void *)v18[2];
      v21 = v18[3] - (_QWORD)v20;
LABEL_10:
      v22 = (const void *)v13[2];
      if ( v21 == v13[3] - (_QWORD)v22 && (!v21 || !memcmp(v22, v20, v21)) )
        return 0;
LABEL_11:
      sub_1412190(*a1, v4);
      return 1;
    }
    v29 = 1;
    v24 = 0;
    while ( v19 != -8 )
    {
      if ( v19 != -16 || v24 )
        v18 = v24;
      v17 = v15 & (v29 + v17);
      v31 = (__int64 *)(v9 + 40LL * v17);
      v19 = *v31;
      if ( a2 == *v31 )
      {
        v20 = (const void *)v31[2];
        v21 = v31[3] - (_QWORD)v20;
        goto LABEL_10;
      }
      ++v29;
      v24 = v18;
      v18 = (__int64 *)(v9 + 40LL * v17);
    }
    v25 = *(_DWORD *)(v7 + 8);
    if ( !v24 )
      v24 = v18;
    ++*(_QWORD *)v7;
    v30 = (v25 >> 1) + 1;
    if ( 4 * v30 < (unsigned int)(3 * v16) )
    {
      if ( (int)v16 - *(_DWORD *)(v7 + 12) - v30 > (unsigned int)v16 >> 3 )
        goto LABEL_17;
      v23 = v16;
LABEL_16:
      sub_1A1C610(v7, v23);
      sub_1A1A140(v7, &v32, &v33);
      v24 = v33;
      v25 = *(_DWORD *)(v7 + 8);
LABEL_17:
      *(_DWORD *)(v7 + 8) = (2 * (v25 >> 1) + 2) | v25 & 1;
      if ( *v24 != -8 )
        --*(_DWORD *)(v7 + 12);
      v26 = v32;
      v24[1] = 0;
      v24[2] = 0;
      *v24 = v26;
      v24[3] = 0;
      v24[4] = 0;
      if ( v13[3] != v13[2] )
        goto LABEL_11;
      return 0;
    }
LABEL_15:
    v23 = 2 * v16;
    goto LABEL_16;
  }
  return result;
}
