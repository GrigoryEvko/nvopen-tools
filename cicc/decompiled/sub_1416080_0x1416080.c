// Function: sub_1416080
// Address: 0x1416080
//
char *__fastcall sub_1416080(__int64 a1, __int64 a2)
{
  __int64 v4; // rbx
  unsigned int v5; // r14d
  unsigned int v6; // ecx
  unsigned int v7; // edx
  __int64 *v8; // rax
  __int64 v9; // rsi
  __int64 v11; // r15
  int v12; // ebx
  int v13; // ebx
  __int64 v14; // r15
  unsigned int v15; // edx
  __int64 *v16; // rax
  __int64 v17; // rsi
  int v18; // eax
  int v19; // r10d
  __int64 v20; // rsi
  int v21; // r10d
  __int64 v22; // r11
  int v23; // ecx
  unsigned int v24; // edx
  __int64 v25; // r9
  int v26; // ecx
  int v27; // r9d
  __int64 *v28; // r8
  int v29; // ecx
  int v30; // edi
  int v31; // r8d
  __int64 *v32; // rdi
  __int64 v33; // [rsp+8h] [rbp-48h]
  __int64 v34; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v35[7]; // [rsp+18h] [rbp-38h] BYREF

  v4 = *(_QWORD *)(a1 + 40);
  v5 = *(_DWORD *)(a1 + 56);
  v34 = a2;
  v33 = v4;
  if ( !v5 )
  {
    v11 = *(_QWORD *)(a2 + 8);
    if ( v11 )
      goto LABEL_6;
    v14 = a1 + 32;
    v13 = 0;
    goto LABEL_23;
  }
  v6 = v5 - 1;
  v7 = (v5 - 1) & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
  v8 = (__int64 *)(v4 + 16LL * v7);
  v9 = *v8;
  if ( a2 != *v8 )
  {
    v18 = 1;
    while ( v9 != -8 )
    {
      v30 = v18 + 1;
      v7 = v6 & (v18 + v7);
      v8 = (__int64 *)(v4 + 16LL * v7);
      v9 = *v8;
      if ( a2 == *v8 )
        goto LABEL_3;
      v18 = v30;
    }
    goto LABEL_20;
  }
LABEL_3:
  if ( v8 == (__int64 *)(v4 + 16LL * v5) )
  {
LABEL_20:
    v11 = *(_QWORD *)(a2 + 8);
    if ( !v11 )
    {
      v14 = a1 + 32;
      v13 = 0;
LABEL_14:
      v15 = v6 & (((unsigned int)a2 >> 9) ^ ((unsigned int)a2 >> 4));
      v16 = (__int64 *)(v33 + 16LL * v15);
      v17 = *v16;
      if ( a2 == *v16 )
      {
LABEL_15:
        *((_DWORD *)v16 + 2) = v13;
        return sub_1415970(a1, a2);
      }
      v27 = 1;
      v28 = 0;
      while ( v17 != -8 )
      {
        if ( !v28 && v17 == -16 )
          v28 = v16;
        v15 = v6 & (v27 + v15);
        v16 = (__int64 *)(v33 + 16LL * v15);
        v17 = *v16;
        if ( a2 == *v16 )
          goto LABEL_15;
        ++v27;
      }
      v29 = *(_DWORD *)(a1 + 48);
      if ( v28 )
        v16 = v28;
      ++*(_QWORD *)(a1 + 32);
      v26 = v29 + 1;
      if ( 4 * v26 < 3 * v5 )
      {
        v20 = a2;
        if ( v5 - *(_DWORD *)(a1 + 52) - v26 > v5 >> 3 )
          goto LABEL_27;
        sub_13FEAC0(v14, v5);
        sub_13FDDE0(v14, &v34, v35);
        v16 = (__int64 *)v35[0];
        v20 = v34;
        v23 = *(_DWORD *)(a1 + 48);
LABEL_26:
        v26 = v23 + 1;
LABEL_27:
        *(_DWORD *)(a1 + 48) = v26;
        if ( *v16 != -8 )
          --*(_DWORD *)(a1 + 52);
        *v16 = v20;
        *((_DWORD *)v16 + 2) = 0;
        goto LABEL_15;
      }
LABEL_24:
      sub_13FEAC0(v14, 2 * v5);
      v19 = *(_DWORD *)(a1 + 56);
      if ( !v19 )
      {
        ++*(_DWORD *)(a1 + 48);
        BUG();
      }
      v20 = v34;
      v21 = v19 - 1;
      v22 = *(_QWORD *)(a1 + 40);
      v23 = *(_DWORD *)(a1 + 48);
      v24 = v21 & (((unsigned int)v34 >> 9) ^ ((unsigned int)v34 >> 4));
      v16 = (__int64 *)(v22 + 16LL * v24);
      v25 = *v16;
      if ( *v16 != v34 )
      {
        v31 = 1;
        v32 = 0;
        while ( v25 != -8 )
        {
          if ( !v32 && v25 == -16 )
            v32 = v16;
          v24 = v21 & (v31 + v24);
          v16 = (__int64 *)(v22 + 16LL * v24);
          v25 = *v16;
          if ( v34 == *v16 )
            goto LABEL_26;
          ++v31;
        }
        v26 = v23 + 1;
        if ( v32 )
          v16 = v32;
        goto LABEL_27;
      }
      goto LABEL_26;
    }
LABEL_6:
    while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v11) + 16) - 25) > 9u )
    {
      v11 = *(_QWORD *)(v11 + 8);
      if ( !v11 )
      {
        v13 = 0;
        goto LABEL_12;
      }
    }
    v12 = 0;
    while ( 1 )
    {
      v11 = *(_QWORD *)(v11 + 8);
      if ( !v11 )
        break;
      while ( (unsigned __int8)(*(_BYTE *)(sub_1648700(v11) + 16) - 25) <= 9u )
      {
        v11 = *(_QWORD *)(v11 + 8);
        ++v12;
        if ( !v11 )
          goto LABEL_11;
      }
    }
LABEL_11:
    v13 = v12 + 1;
LABEL_12:
    v14 = a1 + 32;
    if ( v5 )
    {
      v6 = v5 - 1;
      goto LABEL_14;
    }
LABEL_23:
    ++*(_QWORD *)(a1 + 32);
    goto LABEL_24;
  }
  return sub_1415970(a1, a2);
}
