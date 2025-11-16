// Function: sub_22BE510
// Address: 0x22be510
//
__int64 __fastcall sub_22BE510(__int64 a1)
{
  int v1; // r14d
  __int64 result; // rax
  _QWORD *v3; // rbx
  unsigned int v4; // eax
  __int64 v5; // r13
  _QWORD *v6; // r13
  __int64 v7; // rdx
  _QWORD *v8; // r8
  __int64 v9; // rax
  int v10; // edx
  int v11; // ebx
  unsigned int v12; // r14d
  unsigned int v13; // eax
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  _QWORD *v16; // [rsp+8h] [rbp-98h]
  _QWORD v17[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v18; // [rsp+28h] [rbp-78h]
  __int64 v19; // [rsp+30h] [rbp-70h]
  __int64 (__fastcall **v20)(); // [rsp+40h] [rbp-60h]
  __int64 v21; // [rsp+48h] [rbp-58h] BYREF
  __int64 v22; // [rsp+50h] [rbp-50h]
  __int64 v23; // [rsp+58h] [rbp-48h]
  __int64 v24; // [rsp+60h] [rbp-40h]

  v1 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v1 || (result = *(unsigned int *)(a1 + 20), (_DWORD)result) )
  {
    v3 = *(_QWORD **)(a1 + 8);
    v4 = 4 * v1;
    v5 = 5LL * *(unsigned int *)(a1 + 24);
    if ( (unsigned int)(4 * v1) < 0x40 )
      v4 = 64;
    if ( *(_DWORD *)(a1 + 24) > v4 )
    {
      v17[1] = 0;
      v8 = &v3[v5];
      v20 = off_4A09D90;
      v17[0] = 2;
      v18 = -4096;
      v19 = 0;
      v21 = 2;
      v22 = 0;
      v23 = -8192;
      v24 = 0;
      do
      {
        v9 = v3[3];
        *v3 = &unk_49DB368;
        if ( v9 != 0 && v9 != -4096 && v9 != -8192 )
        {
          v16 = v8;
          sub_BD60C0(v3 + 1);
          v8 = v16;
        }
        v3 += 5;
      }
      while ( v8 != v3 );
      v20 = (__int64 (__fastcall **)())&unk_49DB368;
      if ( v23 != -4096 && v23 != 0 && v23 != -8192 )
        sub_BD60C0(&v21);
      if ( v18 != -4096 && v18 != 0 && v18 != -8192 )
        sub_BD60C0(v17);
      v10 = *(_DWORD *)(a1 + 24);
      if ( v1 )
      {
        v11 = 64;
        v12 = v1 - 1;
        if ( v12 )
        {
          _BitScanReverse(&v13, v12);
          v11 = 1 << (33 - (v13 ^ 0x1F));
          if ( v11 < 64 )
            v11 = 64;
        }
        if ( v11 != v10 )
        {
          sub_C7D6A0(*(_QWORD *)(a1 + 8), v5 * 8, 8);
          v14 = ((((((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
                   | (4 * v11 / 3u + 1)
                   | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
                 | (4 * v11 / 3u + 1)
                 | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
                 | (4 * v11 / 3u + 1)
                 | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
               | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
               | (4 * v11 / 3u + 1)
               | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 16;
          v15 = (v14
               | (((((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
                   | (4 * v11 / 3u + 1)
                   | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
                 | (4 * v11 / 3u + 1)
                 | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
                 | (4 * v11 / 3u + 1)
                 | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
               | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
               | (4 * v11 / 3u + 1)
               | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1))
              + 1;
          *(_DWORD *)(a1 + 24) = v15;
          *(_QWORD *)(a1 + 8) = sub_C7D670(40 * v15, 8);
        }
      }
      else if ( v10 )
      {
        result = sub_C7D6A0(*(_QWORD *)(a1 + 8), v5 * 8, 8);
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
        return result;
      }
      return sub_22BDDA0(a1);
    }
    v6 = &v3[v5];
    v21 = 2;
    v20 = off_4A09D90;
    result = -4096;
    v22 = 0;
    v23 = -4096;
    v24 = 0;
    if ( v6 == v3 )
    {
      *(_QWORD *)(a1 + 16) = 0;
    }
    else
    {
      do
      {
        v7 = v3[3];
        if ( v7 != result )
        {
          if ( v7 != -4096 && v7 != 0 && v7 != -8192 )
          {
            sub_BD60C0(v3 + 1);
            result = v23;
          }
          v3[3] = result;
          if ( result != -4096 && result != 0 && result != -8192 )
            sub_BD6050(v3 + 1, v21 & 0xFFFFFFFFFFFFFFF8LL);
          result = v23;
        }
        v3 += 5;
        *(v3 - 1) = v24;
      }
      while ( v3 != v6 );
      *(_QWORD *)(a1 + 16) = 0;
      v20 = (__int64 (__fastcall **)())&unk_49DB368;
      if ( result != -4096 && result != 0 && result != -8192 )
        return sub_BD60C0(&v21);
    }
  }
  return result;
}
