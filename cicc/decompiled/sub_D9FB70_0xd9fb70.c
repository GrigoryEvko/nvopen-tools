// Function: sub_D9FB70
// Address: 0xd9fb70
//
__int64 __fastcall sub_D9FB70(__int64 a1)
{
  int v1; // ebx
  __int64 result; // rax
  unsigned int v3; // eax
  unsigned int v4; // r15d
  __int64 v5; // rbx
  __int64 v6; // r13
  __int64 v7; // rdx
  _QWORD *v8; // r13
  _QWORD *v9; // rdx
  void *v10; // r9
  __int64 v11; // rax
  unsigned int v12; // eax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rax
  _QWORD *v15; // rbx
  _QWORD *i; // r12
  char v17; // al
  __int64 v18; // rax
  bool v19; // zf
  void *v20; // [rsp+10h] [rbp-A0h]
  _QWORD *v21; // [rsp+18h] [rbp-98h]
  void *v22; // [rsp+20h] [rbp-90h] BYREF
  __int64 v23; // [rsp+28h] [rbp-88h] BYREF
  __int64 v24; // [rsp+38h] [rbp-78h]
  void *v25; // [rsp+50h] [rbp-60h] BYREF
  _QWORD v26[2]; // [rsp+58h] [rbp-58h] BYREF
  __int64 v27; // [rsp+68h] [rbp-48h]
  __int64 v28; // [rsp+70h] [rbp-40h]

  v1 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v1 || (result = *(unsigned int *)(a1 + 20), (_DWORD)result) )
  {
    v3 = 4 * v1;
    v4 = *(_DWORD *)(a1 + 24);
    if ( (unsigned int)(4 * v1) < 0x40 )
      v3 = 64;
    if ( v4 > v3 )
    {
      sub_D982A0(&v22, -4096, 0);
      sub_D982A0(&v25, -8192, 0);
      v8 = *(_QWORD **)(a1 + 8);
      v9 = &v8[6 * *(unsigned int *)(a1 + 24)];
      if ( v8 != v9 )
      {
        v10 = &unk_49DB368;
        do
        {
          v11 = v8[3];
          *v8 = v10;
          if ( v11 != 0 && v11 != -4096 && v11 != -8192 )
          {
            v20 = v10;
            v21 = v9;
            sub_BD60C0(v8 + 1);
            v10 = v20;
            v9 = v21;
          }
          v8 += 6;
        }
        while ( v9 != v8 );
      }
      v25 = &unk_49DB368;
      if ( v27 != -4096 && v27 != 0 && v27 != -8192 )
        sub_BD60C0(v26);
      v22 = &unk_49DB368;
      if ( v24 != -4096 && v24 != 0 && v24 != -8192 )
        sub_BD60C0(&v23);
      if ( v1 )
      {
        v12 = v1 - 1;
        v1 = 64;
        if ( v12 )
        {
          _BitScanReverse(&v12, v12);
          v1 = 1 << (33 - (v12 ^ 0x1F));
          if ( v1 < 64 )
            v1 = 64;
        }
      }
      if ( *(_DWORD *)(a1 + 24) != v1 )
      {
        result = sub_C7D6A0(*(_QWORD *)(a1 + 8), 48LL * v4, 8);
        if ( v1 )
        {
          v13 = ((((((((4 * v1 / 3u + 1) | ((unsigned __int64)(4 * v1 / 3u + 1) >> 1)) >> 2)
                   | (4 * v1 / 3u + 1)
                   | ((unsigned __int64)(4 * v1 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v1 / 3u + 1) | ((unsigned __int64)(4 * v1 / 3u + 1) >> 1)) >> 2)
                 | (4 * v1 / 3u + 1)
                 | ((unsigned __int64)(4 * v1 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v1 / 3u + 1) | ((unsigned __int64)(4 * v1 / 3u + 1) >> 1)) >> 2)
                 | (4 * v1 / 3u + 1)
                 | ((unsigned __int64)(4 * v1 / 3u + 1) >> 1)) >> 4)
               | (((4 * v1 / 3u + 1) | ((unsigned __int64)(4 * v1 / 3u + 1) >> 1)) >> 2)
               | (4 * v1 / 3u + 1)
               | ((unsigned __int64)(4 * v1 / 3u + 1) >> 1)) >> 16;
          v14 = (v13
               | (((((((4 * v1 / 3u + 1) | ((unsigned __int64)(4 * v1 / 3u + 1) >> 1)) >> 2)
                   | (4 * v1 / 3u + 1)
                   | ((unsigned __int64)(4 * v1 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v1 / 3u + 1) | ((unsigned __int64)(4 * v1 / 3u + 1) >> 1)) >> 2)
                 | (4 * v1 / 3u + 1)
                 | ((unsigned __int64)(4 * v1 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v1 / 3u + 1) | ((unsigned __int64)(4 * v1 / 3u + 1) >> 1)) >> 2)
                 | (4 * v1 / 3u + 1)
                 | ((unsigned __int64)(4 * v1 / 3u + 1) >> 1)) >> 4)
               | (((4 * v1 / 3u + 1) | ((unsigned __int64)(4 * v1 / 3u + 1) >> 1)) >> 2)
               | (4 * v1 / 3u + 1)
               | ((unsigned __int64)(4 * v1 / 3u + 1) >> 1))
              + 1;
          *(_DWORD *)(a1 + 24) = v14;
          *(_QWORD *)(a1 + 8) = sub_C7D670(48 * v14, 8);
          return sub_D9FA70(a1);
        }
        else
        {
          *(_QWORD *)(a1 + 8) = 0;
          *(_QWORD *)(a1 + 16) = 0;
          *(_DWORD *)(a1 + 24) = 0;
        }
        return result;
      }
      *(_QWORD *)(a1 + 16) = 0;
      sub_D982A0(&v25, -4096, 0);
      v15 = *(_QWORD **)(a1 + 8);
      for ( i = &v15[6 * *(unsigned int *)(a1 + 24)]; i != v15; v15 += 6 )
      {
        if ( v15 )
        {
          v17 = v26[0];
          v15[2] = 0;
          v15[1] = v17 & 6;
          v18 = v27;
          v19 = v27 == -4096;
          v15[3] = v27;
          if ( v18 != 0 && !v19 && v18 != -8192 )
            sub_BD6050(v15 + 1, v26[0] & 0xFFFFFFFFFFFFFFF8LL);
          *v15 = &unk_49DE910;
          v15[4] = v28;
        }
      }
      v25 = &unk_49DB368;
      result = v27;
    }
    else
    {
      sub_D982A0(&v25, -4096, 0);
      v5 = *(_QWORD *)(a1 + 8);
      result = v27;
      v6 = v5 + 48LL * *(unsigned int *)(a1 + 24);
      if ( v6 == v5 )
      {
        result = v27;
      }
      else
      {
        do
        {
          v7 = *(_QWORD *)(v5 + 24);
          if ( v7 != result )
          {
            if ( v7 != -4096 && v7 != 0 && v7 != -8192 )
            {
              sub_BD60C0((_QWORD *)(v5 + 8));
              result = v27;
            }
            *(_QWORD *)(v5 + 24) = result;
            if ( result != 0 && result != -4096 && result != -8192 )
              sub_BD6050((unsigned __int64 *)(v5 + 8), v26[0] & 0xFFFFFFFFFFFFFFF8LL);
            result = v27;
          }
          v5 += 48;
          *(_QWORD *)(v5 - 16) = v28;
        }
        while ( v5 != v6 );
      }
      *(_QWORD *)(a1 + 16) = 0;
      v25 = &unk_49DB368;
    }
    if ( result != -4096 && result != 0 && result != -8192 )
      return sub_BD60C0(v26);
  }
  return result;
}
