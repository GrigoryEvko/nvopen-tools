// Function: sub_ED5910
// Address: 0xed5910
//
unsigned __int64 *__fastcall sub_ED5910(unsigned __int64 *a1, __int64 *a2, void *a3, size_t a4)
{
  __int64 v7; // rbx
  __int64 v8; // rax
  unsigned int v9; // esi
  __int64 v10; // r13
  __int64 v11; // r9
  __int64 v12; // r8
  __int64 *v13; // rdx
  __int64 v14; // rdi
  int v16; // edx
  int v17; // edx
  __int64 v18; // rsi
  int v19; // edi
  unsigned int v20; // eax
  __int64 *v21; // rcx
  __int64 v22; // r8
  int v23; // r11d
  int v24; // edx
  int v25; // edx
  int v26; // edx
  __int64 *v27; // r9
  __int64 v28; // r8
  int v29; // r10d
  unsigned int v30; // eax
  __int64 v31; // rsi
  int v32; // r10d
  unsigned __int64 v33; // [rsp+8h] [rbp-58h]
  __int64 v34; // [rsp+10h] [rbp-50h]
  __int64 v35; // [rsp+18h] [rbp-48h]
  __int64 v36[7]; // [rsp+28h] [rbp-38h] BYREF

  sub_ED3D70(v36, *a2, a3, a4);
  if ( (v36[0] & 0xFFFFFFFFFFFFFFFELL) != 0 )
  {
    *a1 = v36[0] & 0xFFFFFFFFFFFFFFFELL | 1;
    return a1;
  }
  v7 = *a2;
  v34 = *a2 + 120;
  v35 = a2[1];
  v8 = sub_B2F650((__int64)a3, a4);
  v9 = *(_DWORD *)(v7 + 144);
  v10 = v8;
  if ( !v9 )
  {
    ++*(_QWORD *)(v7 + 120);
    goto LABEL_8;
  }
  v11 = *(_QWORD *)(v7 + 128);
  v12 = ((unsigned int)((0xBF58476D1CE4E5B9LL * v8) >> 31) ^ (484763065 * (_DWORD)v8)) & (v9 - 1);
  v13 = (__int64 *)(v11 + 16 * v12);
  v14 = *v13;
  if ( v8 != *v13 )
  {
    v23 = 1;
    v21 = 0;
    while ( v14 != -1 )
    {
      if ( v21 || v14 != -2 )
        v13 = v21;
      LODWORD(v12) = (v9 - 1) & (v23 + v12);
      v14 = *(_QWORD *)(v11 + 16LL * (unsigned int)v12);
      if ( v8 == v14 )
        goto LABEL_4;
      ++v23;
      v21 = v13;
      v13 = (__int64 *)(v11 + 16LL * (unsigned int)v12);
    }
    if ( !v21 )
      v21 = v13;
    v24 = *(_DWORD *)(v7 + 136);
    ++*(_QWORD *)(v7 + 120);
    v19 = v24 + 1;
    if ( 4 * (v24 + 1) < 3 * v9 )
    {
      if ( v9 - *(_DWORD *)(v7 + 140) - v19 > v9 >> 3 )
        goto LABEL_10;
      v33 = ((0xBF58476D1CE4E5B9LL * v8) >> 31) ^ (0xBF58476D1CE4E5B9LL * v8);
      sub_ED5710(v34, v9);
      v25 = *(_DWORD *)(v7 + 144);
      if ( v25 )
      {
        v26 = v25 - 1;
        v27 = 0;
        v28 = *(_QWORD *)(v7 + 128);
        v29 = 1;
        v30 = v26 & v33;
        v19 = *(_DWORD *)(v7 + 136) + 1;
        v21 = (__int64 *)(v28 + 16LL * (v26 & (unsigned int)v33));
        v31 = *v21;
        if ( v10 != *v21 )
        {
          while ( v31 != -1 )
          {
            if ( !v27 && v31 == -2 )
              v27 = v21;
            v30 = v26 & (v29 + v30);
            v21 = (__int64 *)(v28 + 16LL * v30);
            v31 = *v21;
            if ( v10 == *v21 )
              goto LABEL_10;
            ++v29;
          }
LABEL_22:
          if ( v27 )
            v21 = v27;
          goto LABEL_10;
        }
        goto LABEL_10;
      }
      goto LABEL_42;
    }
LABEL_8:
    sub_ED5710(v34, 2 * v9);
    v16 = *(_DWORD *)(v7 + 144);
    if ( v16 )
    {
      v17 = v16 - 1;
      v18 = *(_QWORD *)(v7 + 128);
      v19 = *(_DWORD *)(v7 + 136) + 1;
      v20 = v17 & (((0xBF58476D1CE4E5B9LL * v10) >> 31) ^ (484763065 * v10));
      v21 = (__int64 *)(v18 + 16LL * v20);
      v22 = *v21;
      if ( v10 != *v21 )
      {
        v32 = 1;
        v27 = 0;
        while ( v22 != -1 )
        {
          if ( !v27 && v22 == -2 )
            v27 = v21;
          v20 = v17 & (v32 + v20);
          v21 = (__int64 *)(v18 + 16LL * v20);
          v22 = *v21;
          if ( v10 == *v21 )
            goto LABEL_10;
          ++v32;
        }
        goto LABEL_22;
      }
LABEL_10:
      *(_DWORD *)(v7 + 136) = v19;
      if ( *v21 != -1 )
        --*(_DWORD *)(v7 + 140);
      *v21 = v10;
      v21[1] = v35;
      goto LABEL_4;
    }
LABEL_42:
    ++*(_DWORD *)(v7 + 136);
    BUG();
  }
LABEL_4:
  *a1 = 1;
  return a1;
}
