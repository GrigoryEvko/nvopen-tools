// Function: sub_22C26A0
// Address: 0x22c26a0
//
__int64 __fastcall sub_22C26A0(__int64 a1)
{
  int v2; // r14d
  __int64 result; // rax
  __int64 v4; // rax
  __int64 v5; // rbx
  unsigned int v6; // ecx
  __int64 v7; // rsi
  unsigned int v8; // eax
  __int64 v9; // rsi
  __int64 v10; // r13
  bool v11; // cc
  __int64 j; // rdx
  __int64 v13; // rax
  bool v14; // zf
  __int64 i; // rcx
  __int64 v16; // rax
  __int64 v17; // rax
  int v18; // edi
  int v19; // ebx
  unsigned int v20; // eax
  unsigned __int64 v21; // rdx
  unsigned __int64 v22; // rax
  _QWORD v23[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v24; // [rsp+28h] [rbp-78h]
  unsigned __int8 v25; // [rsp+30h] [rbp-70h]
  void *v26; // [rsp+40h] [rbp-60h]
  _QWORD v27[2]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v28; // [rsp+58h] [rbp-48h]
  char v29; // [rsp+60h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      return result;
  }
  v4 = *(unsigned int *)(a1 + 24);
  v5 = *(_QWORD *)(a1 + 8);
  v23[0] = 2;
  v23[1] = 0;
  v6 = v4;
  v7 = 3 * v4;
  v8 = 4 * v2;
  v25 = 0;
  v9 = 16 * v7;
  v24 = -4096;
  if ( (unsigned int)(4 * v2) < 0x40 )
    v8 = 64;
  v29 = 0;
  v10 = v5 + v9;
  v27[0] = 2;
  v11 = v6 <= v8;
  v27[1] = 0;
  v28 = -8192;
  result = (__int64)&unk_49DE8C0;
  v26 = &unk_49DE8C0;
  if ( !v11 )
  {
    for ( i = -4096; ; i = v24 )
    {
      v17 = *(_QWORD *)(v5 + 24);
      if ( v17 != i && v17 != v28 )
        sub_22C1BD0((unsigned __int64 *)(v5 + 40));
      if ( *(_BYTE *)(v5 + 32) )
      {
        *(_QWORD *)(v5 + 24) = 0;
        *(_QWORD *)v5 = &unk_49DB368;
      }
      else
      {
        v16 = *(_QWORD *)(v5 + 24);
        *(_QWORD *)v5 = &unk_49DB368;
        if ( v16 != -4096 && v16 != 0 && v16 != -8192 )
          sub_BD60C0((_QWORD *)(v5 + 8));
      }
      v5 += 48;
      if ( v5 == v10 )
        break;
    }
    if ( !v29 )
    {
      v26 = &unk_49DB368;
      if ( v28 != -4096 && v28 != -8192 )
      {
        if ( v28 )
          sub_BD60C0(v27);
      }
    }
    if ( !v25 && v24 != -4096 && v24 != 0 && v24 != -8192 )
      sub_BD60C0(v23);
    v18 = *(_DWORD *)(a1 + 24);
    if ( v2 )
    {
      v19 = 64;
      if ( v2 != 1 )
      {
        _BitScanReverse(&v20, v2 - 1);
        v19 = 1 << (33 - (v20 ^ 0x1F));
        if ( v19 < 64 )
          v19 = 64;
      }
      if ( v19 != v18 )
      {
        sub_C7D6A0(*(_QWORD *)(a1 + 8), v9, 8);
        v21 = ((((((((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
                 | (4 * v19 / 3u + 1)
                 | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 4)
               | (((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
               | (4 * v19 / 3u + 1)
               | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 8)
             | (((((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
               | (4 * v19 / 3u + 1)
               | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 4)
             | (((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
             | (4 * v19 / 3u + 1)
             | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 16;
        v22 = (v21
             | (((((((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
                 | (4 * v19 / 3u + 1)
                 | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 4)
               | (((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
               | (4 * v19 / 3u + 1)
               | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 8)
             | (((((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
               | (4 * v19 / 3u + 1)
               | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 4)
             | (((4 * v19 / 3u + 1) | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1)) >> 2)
             | (4 * v19 / 3u + 1)
             | ((unsigned __int64)(4 * v19 / 3u + 1) >> 1))
            + 1;
        *(_DWORD *)(a1 + 24) = v22;
        *(_QWORD *)(a1 + 8) = sub_C7D670(48 * v22, 8);
      }
    }
    else if ( v18 )
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v9, 8);
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
      return result;
    }
    return sub_22BDEB0(a1);
  }
  if ( v5 == v10 )
    goto LABEL_24;
  for ( j = -4096; ; j = v24 )
  {
    result = *(_QWORD *)(v5 + 24);
    if ( result != j )
    {
      if ( result == v28 )
      {
        if ( !*(_BYTE *)(v5 + 32) )
        {
LABEL_16:
          if ( result != -4096 && result != 0 && result != -8192 )
            sub_BD60C0((_QWORD *)(v5 + 8));
          v13 = v24;
          goto LABEL_20;
        }
      }
      else
      {
        sub_22C1BD0((unsigned __int64 *)(v5 + 40));
        if ( !*(_BYTE *)(v5 + 32) )
        {
          result = *(_QWORD *)(v5 + 24);
          if ( result != v24 )
            goto LABEL_16;
LABEL_10:
          result = v25;
          *(_BYTE *)(v5 + 32) = v25;
          goto LABEL_11;
        }
      }
      *(_QWORD *)(v5 + 24) = 0;
      v13 = v24;
      if ( v24 )
      {
LABEL_20:
        *(_QWORD *)(v5 + 24) = v13;
        if ( v13 != -4096 && v13 != 0 && v13 != -8192 )
          sub_BD6050((unsigned __int64 *)(v5 + 8), v23[0] & 0xFFFFFFFFFFFFFFF8LL);
        goto LABEL_10;
      }
      goto LABEL_10;
    }
LABEL_11:
    v5 += 48;
    if ( v5 == v10 )
      break;
  }
  if ( !v29 )
  {
    result = v28;
    v26 = &unk_49DB368;
    if ( v28 != -4096 && v28 != 0 && v28 != -8192 )
    {
      result = sub_BD60C0(v27);
      v14 = v25 == 0;
      *(_QWORD *)(a1 + 16) = 0;
      if ( v14 )
        goto LABEL_28;
      return result;
    }
  }
LABEL_24:
  *(_QWORD *)(a1 + 16) = 0;
  if ( !v25 )
  {
LABEL_28:
    result = v24;
    if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
      return sub_BD60C0(v23);
  }
  return result;
}
