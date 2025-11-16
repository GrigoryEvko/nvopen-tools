// Function: sub_145FF90
// Address: 0x145ff90
//
void __fastcall sub_145FF90(__int64 a1)
{
  int v1; // ebx
  unsigned int v2; // eax
  __int64 v3; // rbx
  __int64 j; // r13
  __int64 v5; // rax
  __int64 v6; // rdx
  _QWORD *v7; // r13
  _QWORD *i; // r15
  __int64 v9; // rax
  int v10; // r13d
  unsigned int v11; // ebx
  unsigned int v12; // eax
  unsigned __int64 v13; // rdx
  unsigned __int64 v14; // rax
  void *v15; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v16[2]; // [rsp+18h] [rbp-88h] BYREF
  __int64 v17; // [rsp+28h] [rbp-78h]
  __int64 v18; // [rsp+30h] [rbp-70h]
  void *v19; // [rsp+40h] [rbp-60h] BYREF
  _BYTE v20[16]; // [rsp+48h] [rbp-58h] BYREF
  __int64 v21; // [rsp+58h] [rbp-48h]

  v1 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v1 || *(_DWORD *)(a1 + 20) )
  {
    v2 = 4 * v1;
    if ( (unsigned int)(4 * v1) < 0x40 )
      v2 = 64;
    if ( *(_DWORD *)(a1 + 24) > v2 )
    {
      sub_1457D90(&v15, -8, 0);
      sub_1457D90(&v19, -16, 0);
      v7 = *(_QWORD **)(a1 + 8);
      for ( i = &v7[6 * *(unsigned int *)(a1 + 24)]; i != v7; v7 += 6 )
      {
        v9 = v7[3];
        *v7 = &unk_49EE2B0;
        if ( v9 != -8 && v9 != 0 && v9 != -16 )
          sub_1649B30(v7 + 1);
      }
      v19 = &unk_49EE2B0;
      sub_1455FA0((__int64)v20);
      v15 = &unk_49EE2B0;
      sub_1455FA0((__int64)v16);
      if ( v1 )
      {
        v10 = 64;
        v11 = v1 - 1;
        if ( v11 )
        {
          _BitScanReverse(&v12, v11);
          v10 = 1 << (33 - (v12 ^ 0x1F));
          if ( v10 < 64 )
            v10 = 64;
        }
        if ( *(_DWORD *)(a1 + 24) != v10 )
        {
          j___libc_free_0(*(_QWORD *)(a1 + 8));
          v13 = ((((((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
                   | (4 * v10 / 3u + 1)
                   | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
                 | (4 * v10 / 3u + 1)
                 | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
                 | (4 * v10 / 3u + 1)
                 | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
               | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
               | (4 * v10 / 3u + 1)
               | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 16;
          v14 = (v13
               | (((((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
                   | (4 * v10 / 3u + 1)
                   | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
                 | (4 * v10 / 3u + 1)
                 | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
                 | (4 * v10 / 3u + 1)
                 | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 4)
               | (((4 * v10 / 3u + 1) | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1)) >> 2)
               | (4 * v10 / 3u + 1)
               | ((unsigned __int64)(4 * v10 / 3u + 1) >> 1))
              + 1;
          *(_DWORD *)(a1 + 24) = v14;
          *(_QWORD *)(a1 + 8) = sub_22077B0(48 * v14);
        }
      }
      else if ( *(_DWORD *)(a1 + 24) )
      {
        j___libc_free_0(*(_QWORD *)(a1 + 8));
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
        return;
      }
      sub_145FEA0(a1);
      return;
    }
    sub_1457D90(&v15, -8, 0);
    sub_1457D90(&v19, -16, 0);
    v3 = *(_QWORD *)(a1 + 8);
    for ( j = v3 + 48LL * *(unsigned int *)(a1 + 24); j != v3; v3 += 48 )
    {
      v5 = v17;
      v6 = *(_QWORD *)(v3 + 24);
      if ( v6 != v17 )
      {
        if ( v6 != -8 && v6 != 0 && v6 != -16 )
        {
          sub_1649B30(v3 + 8);
          v5 = v17;
        }
        *(_QWORD *)(v3 + 24) = v5;
        if ( v5 != -8 && v5 != 0 && v5 != -16 )
          sub_1649AC0(v3 + 8, v16[0] & 0xFFFFFFFFFFFFFFF8LL);
        *(_QWORD *)(v3 + 32) = v18;
      }
    }
    *(_QWORD *)(a1 + 16) = 0;
    v19 = &unk_49EE2B0;
    if ( v21 != -8 && v21 != 0 && v21 != -16 )
      sub_1649B30(v20);
    v15 = &unk_49EE2B0;
    if ( v17 != -8 && v17 != 0 && v17 != -16 )
      sub_1649B30(v16);
  }
}
