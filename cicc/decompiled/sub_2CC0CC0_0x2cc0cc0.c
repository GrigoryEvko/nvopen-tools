// Function: sub_2CC0CC0
// Address: 0x2cc0cc0
//
__int64 __fastcall sub_2CC0CC0(__int64 a1)
{
  int v2; // r14d
  __int64 result; // rax
  __int64 v4; // rsi
  unsigned int v5; // eax
  _QWORD *v6; // rbx
  unsigned int v7; // edx
  __int64 v8; // rsi
  _QWORD *v9; // r12
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // rax
  __int64 v13; // rdx
  __int64 v14; // rax
  bool v15; // zf
  __int64 v16; // rax
  void *v17; // r8
  __int64 v18; // rcx
  __int64 v19; // rax
  unsigned int v20; // eax
  _QWORD *v21; // r12
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 v25; // rdx
  char v26; // cl
  _QWORD *v27; // rbx
  char v28; // al
  __int64 v29; // rax
  void *v30; // [rsp+0h] [rbp-A0h]
  void *v31; // [rsp+0h] [rbp-A0h]
  _QWORD v32[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v33; // [rsp+28h] [rbp-78h]
  __int64 v34; // [rsp+30h] [rbp-70h]
  void *v35; // [rsp+40h] [rbp-60h]
  __int64 v36; // [rsp+48h] [rbp-58h] BYREF
  __int64 v37; // [rsp+50h] [rbp-50h]
  __int64 v38; // [rsp+58h] [rbp-48h]
  __int64 i; // [rsp+60h] [rbp-40h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( !v2 )
  {
    result = *(unsigned int *)(a1 + 20);
    if ( !(_DWORD)result )
      return result;
  }
  v4 = *(unsigned int *)(a1 + 24);
  v5 = 4 * v2;
  v6 = *(_QWORD **)(a1 + 8);
  v32[0] = 2;
  v7 = v4;
  v8 = v4 << 6;
  v32[1] = 0;
  if ( (unsigned int)(4 * v2) < 0x40 )
    v5 = 64;
  v33 = -4096;
  v9 = (_QWORD *)((char *)v6 + v8);
  v34 = 0;
  v36 = 2;
  v37 = 0;
  v38 = -8192;
  v35 = &unk_49DD7B0;
  i = 0;
  if ( v7 > v5 )
  {
    v16 = -4096;
    v17 = &unk_49DB368;
    while ( 1 )
    {
      v18 = v6[3];
      if ( v18 != v16 )
      {
        v16 = v38;
        if ( v18 != v38 )
        {
          v19 = v6[7];
          if ( v19 != -4096 && v19 != 0 && v19 != -8192 )
          {
            v30 = v17;
            sub_BD60C0(v6 + 5);
            v18 = v6[3];
            v17 = v30;
          }
          v16 = v18;
        }
      }
      *v6 = v17;
      if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
      {
        v31 = v17;
        sub_BD60C0(v6 + 1);
        v17 = v31;
      }
      v6 += 8;
      if ( v6 == v9 )
        break;
      v16 = v33;
    }
    v35 = &unk_49DB368;
    if ( v38 != 0 && v38 != -4096 && v38 != -8192 )
      sub_BD60C0(&v36);
    if ( v33 != 0 && v33 != -4096 && v33 != -8192 )
      sub_BD60C0(v32);
    if ( v2 )
    {
      v20 = v2 - 1;
      v2 = 64;
      if ( v20 )
      {
        _BitScanReverse(&v20, v20);
        v2 = 1 << (33 - (v20 ^ 0x1F));
        if ( v2 < 64 )
          v2 = 64;
      }
    }
    v21 = *(_QWORD **)(a1 + 8);
    if ( *(_DWORD *)(a1 + 24) == v2 )
    {
      result = (__int64)&unk_49DD7A0;
      *(_QWORD *)(a1 + 16) = 0;
      v36 = 2;
      v27 = &v21[8 * (unsigned __int64)(unsigned int)v2];
      v37 = 0;
      v38 = -4096;
      v35 = &unk_49DD7B0;
      i = 0;
      if ( v27 != v21 )
      {
        do
        {
          if ( v21 )
          {
            v28 = v36;
            v21[2] = 0;
            v21[1] = v28 & 6;
            v29 = v38;
            v15 = v38 == 0;
            v21[3] = v38;
            if ( v29 != -4096 && !v15 && v29 != -8192 )
              sub_BD6050(v21 + 1, v36 & 0xFFFFFFFFFFFFFFF8LL);
            *v21 = &unk_49DD7B0;
            v21[4] = i;
          }
          v21 += 8;
        }
        while ( v27 != v21 );
        result = v38;
        v35 = &unk_49DB368;
        if ( v38 != 0 && v38 != -4096 && v38 != -8192 )
          return sub_BD60C0(&v36);
      }
    }
    else
    {
      result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v8, 8);
      if ( v2 )
      {
        v22 = ((((((((4 * v2 / 3u + 1) | ((unsigned __int64)(4 * v2 / 3u + 1) >> 1)) >> 2)
                 | (4 * v2 / 3u + 1)
                 | ((unsigned __int64)(4 * v2 / 3u + 1) >> 1)) >> 4)
               | (((4 * v2 / 3u + 1) | ((unsigned __int64)(4 * v2 / 3u + 1) >> 1)) >> 2)
               | (4 * v2 / 3u + 1)
               | ((unsigned __int64)(4 * v2 / 3u + 1) >> 1)) >> 8)
             | (((((4 * v2 / 3u + 1) | ((unsigned __int64)(4 * v2 / 3u + 1) >> 1)) >> 2)
               | (4 * v2 / 3u + 1)
               | ((unsigned __int64)(4 * v2 / 3u + 1) >> 1)) >> 4)
             | (((4 * v2 / 3u + 1) | ((unsigned __int64)(4 * v2 / 3u + 1) >> 1)) >> 2)
             | (4 * v2 / 3u + 1)
             | ((unsigned __int64)(4 * v2 / 3u + 1) >> 1)) >> 16;
        v23 = (v22
             | (((((((4 * v2 / 3u + 1) | ((unsigned __int64)(4 * v2 / 3u + 1) >> 1)) >> 2)
                 | (4 * v2 / 3u + 1)
                 | ((unsigned __int64)(4 * v2 / 3u + 1) >> 1)) >> 4)
               | (((4 * v2 / 3u + 1) | ((unsigned __int64)(4 * v2 / 3u + 1) >> 1)) >> 2)
               | (4 * v2 / 3u + 1)
               | ((unsigned __int64)(4 * v2 / 3u + 1) >> 1)) >> 8)
             | (((((4 * v2 / 3u + 1) | ((unsigned __int64)(4 * v2 / 3u + 1) >> 1)) >> 2)
               | (4 * v2 / 3u + 1)
               | ((unsigned __int64)(4 * v2 / 3u + 1) >> 1)) >> 4)
             | (((4 * v2 / 3u + 1) | ((unsigned __int64)(4 * v2 / 3u + 1) >> 1)) >> 2)
             | (4 * v2 / 3u + 1)
             | ((unsigned __int64)(4 * v2 / 3u + 1) >> 1))
            + 1;
        *(_DWORD *)(a1 + 24) = v23;
        result = sub_C7D670(v23 << 6, 8);
        v24 = *(unsigned int *)(a1 + 24);
        *(_QWORD *)(a1 + 16) = 0;
        *(_QWORD *)(a1 + 8) = result;
        v36 = 2;
        v25 = result + (v24 << 6);
        for ( i = 0; v25 != result; result += 64 )
        {
          if ( result )
          {
            v26 = v36;
            *(_QWORD *)(result + 16) = 0;
            *(_QWORD *)(result + 24) = -4096;
            *(_QWORD *)result = &unk_49DD7B0;
            *(_QWORD *)(result + 8) = v26 & 6;
            *(_QWORD *)(result + 32) = i;
          }
        }
      }
      else
      {
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
      }
    }
    return result;
  }
  v10 = -8192;
  v11 = -4096;
  if ( v6 == v9 )
    goto LABEL_25;
  while ( 1 )
  {
    v13 = v6[3];
    if ( v13 != v11 )
    {
      if ( v13 != v10 )
      {
        v12 = v6[7];
        if ( v12 == -4096 || v12 == 0 || v12 == -8192 )
        {
          v10 = v6[3];
        }
        else
        {
          sub_BD60C0(v6 + 5);
          v10 = v6[3];
          if ( v10 == v33 )
          {
LABEL_11:
            v6[4] = v34;
            v10 = v38;
            goto LABEL_12;
          }
        }
      }
      if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
        sub_BD60C0(v6 + 1);
      v14 = v33;
      v15 = v33 == 0;
      v6[3] = v33;
      if ( v14 != -4096 && !v15 && v14 != -8192 )
        sub_BD6050(v6 + 1, v32[0] & 0xFFFFFFFFFFFFFFF8LL);
      goto LABEL_11;
    }
LABEL_12:
    v6 += 8;
    if ( v6 == v9 )
      break;
    v11 = v33;
  }
  v35 = &unk_49DB368;
  if ( v10 != -4096 && v10 != 0 && v10 != -8192 )
    sub_BD60C0(&v36);
LABEL_25:
  *(_QWORD *)(a1 + 16) = 0;
  result = v33;
  if ( v33 != -4096 && v33 != 0 && v33 != -8192 )
    return sub_BD60C0(v32);
  return result;
}
