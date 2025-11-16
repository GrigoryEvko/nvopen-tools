// Function: sub_DB8730
// Address: 0xdb8730
//
void __fastcall sub_DB8730(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5, __int64 a6, __int64 *a7)
{
  __int64 v7; // r8
  __int64 v9; // rdx
  char v10; // cl
  __int64 v11; // r9
  __int64 v12; // rsi
  unsigned int v13; // eax
  _QWORD *v14; // r10
  __int64 v15; // rdx
  unsigned int v16; // eax
  _QWORD *v17; // rdi
  __int64 v18; // rdx
  __int64 v19; // r8
  __int64 v20; // rcx
  __int64 v21; // rcx
  __int64 v22; // rdx
  __int64 v23; // rdx
  int v24; // r11d
  unsigned int v25; // eax
  __int64 v26; // rdx
  unsigned int v27; // eax
  __int64 v28; // rdx
  int v29; // r10d
  int v30; // ecx
  int v31; // ecx
  int v32; // r10d
  unsigned __int64 v33; // [rsp+0h] [rbp-80h]
  __int64 v34; // [rsp+8h] [rbp-78h]
  __int64 v35; // [rsp+10h] [rbp-70h]
  __int64 v36; // [rsp+18h] [rbp-68h]
  char v37; // [rsp+20h] [rbp-60h]
  char *v38; // [rsp+28h] [rbp-58h] BYREF
  __int64 v39; // [rsp+30h] [rbp-50h]
  _BYTE v40[72]; // [rsp+38h] [rbp-48h] BYREF

  v7 = 4LL * a5;
  v33 = v7 | a3 & 0xFFFFFFFFFFFFFFFBLL;
  v38 = v40;
  v9 = *((unsigned int *)a7 + 10);
  v34 = *a7;
  v35 = a7[1];
  v36 = a7[2];
  v37 = *((_BYTE *)a7 + 24);
  v39 = 0x400000000LL;
  if ( (_DWORD)v9 )
    sub_D915C0((__int64)&v38, (__int64)(a7 + 4), v9, a4, v7, a6);
  v10 = *(_BYTE *)(a1 + 8) & 1;
  if ( v10 )
  {
    v11 = a1 + 16;
    v12 = 3;
  }
  else
  {
    v12 = *(unsigned int *)(a1 + 24);
    v11 = *(_QWORD *)(a1 + 16);
    if ( !(_DWORD)v12 )
    {
      v16 = *(_DWORD *)(a1 + 8);
      ++*(_QWORD *)a1;
      v17 = 0;
      v18 = (v16 >> 1) + 1;
LABEL_12:
      v19 = (unsigned int)(3 * v12);
      goto LABEL_13;
    }
    v12 = (unsigned int)(v12 - 1);
  }
  v13 = v12 & (v33 ^ (v33 >> 9));
  v14 = (_QWORD *)(v11 + 88LL * v13);
  v15 = *v14;
  if ( *v14 == v33 )
    goto LABEL_6;
  v24 = 1;
  v17 = 0;
  while ( v15 != -4 )
  {
    if ( v15 != -16 || v17 )
      v14 = v17;
    v13 = v12 & (v24 + v13);
    v15 = *(_QWORD *)(v11 + 88LL * v13);
    if ( v33 == v15 )
      goto LABEL_6;
    ++v24;
    v17 = v14;
    v14 = (_QWORD *)(v11 + 88LL * v13);
  }
  v16 = *(_DWORD *)(a1 + 8);
  if ( !v17 )
    v17 = v14;
  ++*(_QWORD *)a1;
  v18 = (v16 >> 1) + 1;
  if ( !v10 )
  {
    v12 = *(unsigned int *)(a1 + 24);
    goto LABEL_12;
  }
  v19 = 12;
  v12 = 4;
LABEL_13:
  v20 = (unsigned int)(4 * v18);
  if ( (unsigned int)v20 >= (unsigned int)v19 )
  {
    sub_DB8330(a1, (unsigned int)(2 * v12), v18, v20, v19, v11);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v19 = a1 + 16;
      v21 = 3;
    }
    else
    {
      v30 = *(_DWORD *)(a1 + 24);
      v19 = *(_QWORD *)(a1 + 16);
      if ( !v30 )
        goto LABEL_58;
      v21 = (unsigned int)(v30 - 1);
    }
    v12 = v33;
    v25 = v21 & (v33 ^ (v33 >> 9));
    v17 = (_QWORD *)(v19 + 88LL * v25);
    v26 = *v17;
    if ( *v17 != v33 )
    {
      v32 = 1;
      v11 = 0;
      while ( v26 != -4 )
      {
        if ( v26 == -16 && !v11 )
          v11 = (__int64)v17;
        v25 = v21 & (v32 + v25);
        v17 = (_QWORD *)(v19 + 88LL * v25);
        v26 = *v17;
        if ( v33 == *v17 )
          goto LABEL_28;
        ++v32;
      }
      goto LABEL_34;
    }
LABEL_28:
    v16 = *(_DWORD *)(a1 + 8);
    goto LABEL_15;
  }
  v21 = (unsigned int)(v12 - *(_DWORD *)(a1 + 12) - v18);
  v22 = (unsigned int)v12 >> 3;
  if ( (unsigned int)v21 <= (unsigned int)v22 )
  {
    sub_DB8330(a1, v12, v22, v21, v19, v11);
    if ( (*(_BYTE *)(a1 + 8) & 1) != 0 )
    {
      v19 = a1 + 16;
      v21 = 3;
      goto LABEL_31;
    }
    v31 = *(_DWORD *)(a1 + 24);
    v19 = *(_QWORD *)(a1 + 16);
    if ( v31 )
    {
      v21 = (unsigned int)(v31 - 1);
LABEL_31:
      v12 = v33;
      v27 = v21 & (v33 ^ (v33 >> 9));
      v17 = (_QWORD *)(v19 + 88LL * v27);
      v28 = *v17;
      if ( *v17 != v33 )
      {
        v29 = 1;
        v11 = 0;
        while ( v28 != -4 )
        {
          if ( !v11 && v28 == -16 )
            v11 = (__int64)v17;
          v27 = v21 & (v29 + v27);
          v17 = (_QWORD *)(v19 + 88LL * v27);
          v28 = *v17;
          if ( v33 == *v17 )
            goto LABEL_28;
          ++v29;
        }
LABEL_34:
        if ( v11 )
          v17 = (_QWORD *)v11;
        goto LABEL_28;
      }
      goto LABEL_28;
    }
LABEL_58:
    *(_DWORD *)(a1 + 8) = (2 * (*(_DWORD *)(a1 + 8) >> 1) + 2) | *(_DWORD *)(a1 + 8) & 1;
    BUG();
  }
LABEL_15:
  v23 = 2 * (v16 >> 1) + 2;
  *(_DWORD *)(a1 + 8) = v23 | v16 & 1;
  if ( *v17 != -4 )
    --*(_DWORD *)(a1 + 12);
  *v17 = v33;
  v17[1] = v34;
  v17[2] = v35;
  v17[3] = v36;
  *((_BYTE *)v17 + 32) = v37;
  v17[5] = v17 + 7;
  v17[6] = 0x400000000LL;
  if ( (_DWORD)v39 )
  {
    v12 = (__int64)&v38;
    sub_D91460((__int64)(v17 + 5), &v38, v23, v21, v19, v11);
  }
LABEL_6:
  if ( v38 != v40 )
    _libc_free(v38, v12);
}
