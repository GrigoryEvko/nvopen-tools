// Function: sub_2396AE0
// Address: 0x2396ae0
//
void **__fastcall sub_2396AE0(__int64 a1)
{
  void *v2; // rax
  char v3; // r8
  __int64 v4; // r8
  __int64 v5; // rdi
  int v6; // esi
  unsigned int v7; // ecx
  __int64 *v8; // rdx
  __int64 v9; // r9
  __int64 *v10; // rbx
  unsigned __int64 v11; // r12
  void **result; // rax
  __int64 v13; // rcx
  void **v14; // rsi
  __int64 v15; // rdx
  __int64 v16; // rcx
  int v17; // edx
  unsigned int v18; // esi
  unsigned int v19; // edx
  int v20; // ecx
  __int64 *v21; // rdx
  __int64 *v22; // rsi
  __int64 v23; // rax
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // r9
  int v27; // r11d
  __int64 *v28; // r10
  void *v29; // [rsp+0h] [rbp-20h] BYREF
  __int64 *v30; // [rsp+8h] [rbp-18h] BYREF

  v2 = &unk_4F86B78;
  v3 = *(_BYTE *)(a1 + 16);
  v29 = &unk_4F86B78;
  v4 = v3 & 1;
  if ( (_DWORD)v4 )
  {
    v5 = a1 + 24;
    v6 = 1;
  }
  else
  {
    v18 = *(_DWORD *)(a1 + 32);
    v5 = *(_QWORD *)(a1 + 24);
    if ( !v18 )
    {
      v19 = *(_DWORD *)(a1 + 16);
      ++*(_QWORD *)(a1 + 8);
      v30 = 0;
      v20 = (v19 >> 1) + 1;
      goto LABEL_22;
    }
    v6 = v18 - 1;
  }
  v7 = v6 & (((unsigned int)&unk_4F86B78 >> 9) ^ ((unsigned int)&unk_4F86B78 >> 4));
  v8 = (__int64 *)(v5 + 16LL * v7);
  v9 = *v8;
  if ( (_UNKNOWN *)*v8 != &unk_4F86B78 )
  {
    v27 = 1;
    v28 = 0;
    while ( v9 != -4096 )
    {
      if ( !v28 && v9 == -8192 )
        v28 = v8;
      v7 = v6 & (v27 + v7);
      v8 = (__int64 *)(v5 + 16LL * v7);
      v9 = *v8;
      if ( (_UNKNOWN *)*v8 == &unk_4F86B78 )
        goto LABEL_4;
      ++v27;
    }
    v9 = 6;
    v18 = 2;
    if ( !v28 )
      v28 = v8;
    v19 = *(_DWORD *)(a1 + 16);
    ++*(_QWORD *)(a1 + 8);
    v30 = v28;
    v20 = (v19 >> 1) + 1;
    if ( (_BYTE)v4 )
    {
LABEL_23:
      if ( 4 * v20 >= (unsigned int)v9 )
      {
        v18 *= 2;
      }
      else if ( v18 - *(_DWORD *)(a1 + 20) - v20 > v18 >> 3 )
      {
LABEL_25:
        *(_DWORD *)(a1 + 16) = (2 * (v19 >> 1) + 2) | v19 & 1;
        v21 = v30;
        if ( *v30 != -4096 )
          --*(_DWORD *)(a1 + 20);
        v21[1] = 0;
        v10 = v21 + 1;
        *v21 = (__int64)v2;
        v22 = v21 + 1;
        result = (void **)(v21 + 1);
        goto LABEL_28;
      }
      sub_2396760(a1 + 8, v18);
      sub_2361AB0(a1 + 8, (__int64 *)&v29, &v30);
      v2 = v29;
      v19 = *(_DWORD *)(a1 + 16);
      goto LABEL_25;
    }
    v18 = *(_DWORD *)(a1 + 32);
LABEL_22:
    v9 = 3 * v18;
    goto LABEL_23;
  }
LABEL_4:
  v10 = v8 + 1;
  v11 = v8[1] & 0xFFFFFFFFFFFFFFF8LL;
  if ( (v8[1] & 4) == 0 )
  {
    result = (void **)(v8 + 1);
    if ( v11 )
    {
      v22 = v8 + 2;
      result = (void **)(v8 + 2);
      if ( (_UNKNOWN *)v8[1] == &unk_4F86540 )
        result = (void **)(v8 + 1);
LABEL_29:
      v17 = 0;
      if ( result != (void **)v22 )
        return result;
      goto LABEL_14;
    }
    v22 = v8 + 1;
LABEL_28:
    v11 = 0;
    goto LABEL_29;
  }
  result = *(void ***)v11;
  v13 = 8LL * *(unsigned int *)(v11 + 8);
  v14 = (void **)(*(_QWORD *)v11 + v13);
  v15 = v13 >> 3;
  v16 = v13 >> 5;
  if ( !v16 )
  {
LABEL_39:
    if ( v15 != 2 )
    {
      if ( v15 != 3 )
      {
        if ( v15 != 1 )
        {
LABEL_65:
          result = v14;
          goto LABEL_12;
        }
LABEL_42:
        if ( *result == &unk_4F86540 )
          goto LABEL_12;
        goto LABEL_65;
      }
      if ( *result == &unk_4F86540 )
        goto LABEL_12;
      ++result;
    }
    if ( *result == &unk_4F86540 )
      goto LABEL_12;
    ++result;
    goto LABEL_42;
  }
  while ( *result != &unk_4F86540 )
  {
    if ( result[1] == &unk_4F86540 )
    {
      ++result;
      v17 = 1;
      goto LABEL_13;
    }
    if ( result[2] == &unk_4F86540 )
    {
      result += 2;
      v17 = 1;
      goto LABEL_13;
    }
    if ( result[3] == &unk_4F86540 )
    {
      result += 3;
      v17 = 1;
      goto LABEL_13;
    }
    result += 4;
    if ( !--v16 )
    {
      v15 = v14 - result;
      goto LABEL_39;
    }
  }
LABEL_12:
  v17 = 1;
LABEL_13:
  if ( result == v14 )
  {
LABEL_14:
    if ( v11 )
    {
      if ( v17 != 1 )
      {
        v23 = sub_22077B0(0x30u);
        if ( v23 )
        {
          *(_QWORD *)v23 = v23 + 16;
          *(_QWORD *)(v23 + 8) = 0x400000000LL;
        }
        *v10 = v23 | 4;
        sub_2361B80(v23 & 0xFFFFFFFFFFFFFFF8LL, v11, v23 | 4, v24, v25, v26);
        v11 = *v10 & 0xFFFFFFFFFFFFFFF8LL;
      }
      result = (void **)*(unsigned int *)(v11 + 8);
      if ( (unsigned __int64)result + 1 > *(unsigned int *)(v11 + 12) )
      {
        sub_C8D5F0(v11, (const void *)(v11 + 16), (unsigned __int64)result + 1, 8u, v4, v9);
        result = (void **)*(unsigned int *)(v11 + 8);
      }
      *(_QWORD *)(*(_QWORD *)v11 + 8LL * (_QWORD)result) = &unk_4F86540;
      ++*(_DWORD *)(v11 + 8);
    }
    else
    {
      *v10 = (__int64)&unk_4F86540;
      return (void **)&unk_4F86540;
    }
  }
  return result;
}
