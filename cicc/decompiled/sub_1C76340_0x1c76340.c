// Function: sub_1C76340
// Address: 0x1c76340
//
__int64 __fastcall sub_1C76340(__int64 a1)
{
  int v2; // r14d
  __int64 result; // rax
  __int64 v4; // r12
  _QWORD *v5; // rbx
  unsigned int v6; // eax
  unsigned int v7; // edx
  _QWORD *v8; // r12
  __int64 v9; // rax
  __int64 j; // rcx
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // rax
  bool v14; // zf
  __int64 v15; // rax
  void *v16; // r8
  __int64 v17; // rcx
  __int64 v18; // rax
  int v19; // esi
  __int64 v20; // rbx
  unsigned int v21; // r14d
  unsigned int v22; // eax
  _QWORD *v23; // r12
  unsigned __int64 v24; // rax
  unsigned __int64 v25; // rdi
  __int64 v26; // rdx
  __int64 v27; // rdx
  char v28; // cl
  _QWORD *v29; // rbx
  char v30; // al
  __int64 v31; // rax
  void *v32; // [rsp+8h] [rbp-98h]
  void *v33; // [rsp+8h] [rbp-98h]
  _QWORD v34[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v35; // [rsp+28h] [rbp-78h]
  __int64 v36; // [rsp+30h] [rbp-70h]
  void *v37; // [rsp+40h] [rbp-60h]
  __int64 v38; // [rsp+48h] [rbp-58h] BYREF
  __int64 v39; // [rsp+50h] [rbp-50h]
  __int64 v40; // [rsp+58h] [rbp-48h]
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
  v5 = *(_QWORD **)(a1 + 8);
  v6 = 4 * v2;
  v34[0] = 2;
  v7 = v4;
  v34[1] = 0;
  v8 = &v5[8 * v4];
  v35 = -8;
  if ( (unsigned int)(4 * v2) < 0x40 )
    v6 = 64;
  v36 = 0;
  v38 = 2;
  v39 = 0;
  v40 = -16;
  i = 0;
  v37 = &unk_49E6B50;
  if ( v7 > v6 )
  {
    v15 = -8;
    v16 = &unk_49EE2B0;
    while ( 1 )
    {
      v17 = v5[3];
      if ( v17 != v15 )
      {
        v15 = v40;
        if ( v17 != v40 )
        {
          v18 = v5[7];
          if ( v18 != -8 && v18 != 0 && v18 != -16 )
          {
            v32 = v16;
            sub_1649B30(v5 + 5);
            v17 = v5[3];
            v16 = v32;
          }
          v15 = v17;
        }
      }
      *v5 = v16;
      if ( v15 != -8 && v15 != 0 && v15 != -16 )
      {
        v33 = v16;
        sub_1649B30(v5 + 1);
        v16 = v33;
      }
      v5 += 8;
      if ( v5 == v8 )
        break;
      v15 = v35;
    }
    v37 = &unk_49EE2B0;
    if ( v40 != -8 && v40 != 0 && v40 != -16 )
      sub_1649B30(&v38);
    result = v35;
    if ( v35 != -8 && v35 != 0 && v35 != -16 )
      result = sub_1649B30(v34);
    v19 = *(_DWORD *)(a1 + 24);
    if ( v2 )
    {
      v20 = 64;
      v21 = v2 - 1;
      if ( v21 )
      {
        _BitScanReverse(&v22, v21);
        v20 = (unsigned int)(1 << (33 - (v22 ^ 0x1F)));
        if ( (int)v20 < 64 )
          v20 = 64;
      }
      v23 = *(_QWORD **)(a1 + 8);
      if ( (_DWORD)v20 == v19 )
      {
        *(_QWORD *)(a1 + 16) = 0;
        v38 = 2;
        v29 = &v23[8 * v20];
        v39 = 0;
        v40 = -8;
        v37 = &unk_49E6B50;
        i = 0;
        do
        {
          if ( v23 )
          {
            v30 = v38;
            v23[2] = 0;
            v23[1] = v30 & 6;
            v31 = v40;
            v14 = v40 == -8;
            v23[3] = v40;
            if ( v31 != 0 && !v14 && v31 != -16 )
              sub_1649AC0(v23 + 1, v38 & 0xFFFFFFFFFFFFFFF8LL);
            *v23 = &unk_49E6B50;
            v23[4] = i;
          }
          v23 += 8;
        }
        while ( v29 != v23 );
        result = v40;
        v37 = &unk_49EE2B0;
        if ( v40 != -8 && v40 != 0 && v40 != -16 )
          return sub_1649B30(&v38);
      }
      else
      {
        j___libc_free_0(*(_QWORD *)(a1 + 8));
        v24 = ((((((((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v20 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v20 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 8)
             | (((((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v20 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v20 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 16;
        v25 = (v24
             | (((((((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
                 | (4 * (int)v20 / 3u + 1)
                 | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 4)
               | (((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v20 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 8)
             | (((((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
               | (4 * (int)v20 / 3u + 1)
               | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 4)
             | (((4 * (int)v20 / 3u + 1) | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1)) >> 2)
             | (4 * (int)v20 / 3u + 1)
             | ((unsigned __int64)(4 * (int)v20 / 3u + 1) >> 1))
            + 1;
        *(_DWORD *)(a1 + 24) = v25;
        result = sub_22077B0(v25 << 6);
        v26 = *(unsigned int *)(a1 + 24);
        *(_QWORD *)(a1 + 16) = 0;
        *(_QWORD *)(a1 + 8) = result;
        v38 = 2;
        v27 = result + (v26 << 6);
        for ( i = 0; v27 != result; result += 64 )
        {
          if ( result )
          {
            v28 = v38;
            *(_QWORD *)(result + 16) = 0;
            *(_QWORD *)(result + 24) = -8;
            *(_QWORD *)result = &unk_49E6B50;
            *(_QWORD *)(result + 8) = v28 & 6;
            *(_QWORD *)(result + 32) = i;
          }
        }
      }
    }
    else if ( v19 )
    {
      result = j___libc_free_0(*(_QWORD *)(a1 + 8));
      *(_QWORD *)(a1 + 8) = 0;
      *(_QWORD *)(a1 + 16) = 0;
      *(_DWORD *)(a1 + 24) = 0;
    }
    else
    {
      *(_QWORD *)(a1 + 16) = 0;
    }
    return result;
  }
  if ( v5 == v8 )
  {
    *(_QWORD *)(a1 + 16) = 0;
    goto LABEL_25;
  }
  v9 = -16;
  for ( j = -8; ; j = v35 )
  {
    v12 = v5[3];
    if ( v12 != j )
    {
      if ( v12 != v9 )
      {
        v11 = v5[7];
        if ( v11 == -8 || v11 == 0 || v11 == -16 )
        {
          v9 = v5[3];
        }
        else
        {
          sub_1649B30(v5 + 5);
          v9 = v5[3];
          if ( v9 == v35 )
          {
LABEL_11:
            v5[4] = v36;
            v9 = v40;
            goto LABEL_12;
          }
        }
      }
      if ( v9 != 0 && v9 != -8 && v9 != -16 )
        sub_1649B30(v5 + 1);
      v13 = v35;
      v14 = v35 == 0;
      v5[3] = v35;
      if ( v13 != -8 && !v14 && v13 != -16 )
        sub_1649AC0(v5 + 1, v34[0] & 0xFFFFFFFFFFFFFFF8LL);
      goto LABEL_11;
    }
LABEL_12:
    v5 += 8;
    if ( v5 == v8 )
      break;
  }
  *(_QWORD *)(a1 + 16) = 0;
  v37 = &unk_49EE2B0;
  if ( v9 != -8 && v9 != 0 && v9 != -16 )
    sub_1649B30(&v38);
LABEL_25:
  result = v35;
  if ( v35 != 0 && v35 != -8 && v35 != -16 )
    return sub_1649B30(v34);
  return result;
}
