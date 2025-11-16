// Function: sub_3372D50
// Address: 0x3372d50
//
__int64 __fastcall sub_3372D50(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rdx
  int v7; // r15d
  __int64 v8; // rax
  _QWORD *v9; // rbx
  unsigned int v10; // eax
  __int64 v11; // r14
  _QWORD *v12; // r13
  unsigned __int64 v13; // rdi
  _QWORD *v14; // rbx
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 result; // rax
  unsigned __int64 v18; // rdi
  __int64 v19; // rdx
  int v20; // ebx
  unsigned int v21; // r15d
  unsigned int v22; // eax
  _QWORD *v23; // rdi
  unsigned __int64 v24; // rdx
  unsigned __int64 v25; // rax
  _QWORD *v26; // rax
  __int64 v27; // rdx
  _QWORD *i; // rdx
  _QWORD *v29; // rax

  *(_QWORD *)(a1 + 872) = a3;
  v6 = *(_QWORD *)(a1 + 864);
  *(_QWORD *)(a1 + 880) = a4;
  v7 = *(_DWORD *)(a1 + 1000);
  *(_QWORD *)(a1 + 976) = a2;
  *(_QWORD *)(a1 + 888) = a5;
  v8 = *(_QWORD *)(v6 + 64);
  ++*(_QWORD *)(a1 + 984);
  *(_QWORD *)(a1 + 1024) = v8;
  if ( v7 || *(_DWORD *)(a1 + 1004) )
  {
    v9 = *(_QWORD **)(a1 + 992);
    v10 = 4 * v7;
    v11 = 40LL * *(unsigned int *)(a1 + 1008);
    if ( (unsigned int)(4 * v7) < 0x40 )
      v10 = 64;
    v12 = &v9[(unsigned __int64)v11 / 8];
    if ( *(_DWORD *)(a1 + 1008) > v10 )
    {
      do
      {
        if ( *v9 != -8192 && *v9 != -4096 )
        {
          v18 = v9[1];
          if ( (_QWORD *)v18 != v9 + 3 )
            _libc_free(v18);
        }
        v9 += 5;
      }
      while ( v9 != v12 );
      v19 = *(unsigned int *)(a1 + 1008);
      if ( v7 )
      {
        v20 = 64;
        v21 = v7 - 1;
        if ( v21 )
        {
          _BitScanReverse(&v22, v21);
          v20 = 1 << (33 - (v22 ^ 0x1F));
          if ( v20 < 64 )
            v20 = 64;
        }
        v23 = *(_QWORD **)(a1 + 992);
        if ( (_DWORD)v19 == v20 )
        {
          *(_QWORD *)(a1 + 1000) = 0;
          v29 = &v23[5 * v19];
          do
          {
            if ( v23 )
              *v23 = -4096;
            v23 += 5;
          }
          while ( v29 != v23 );
        }
        else
        {
          sub_C7D6A0((__int64)v23, v11, 8);
          v24 = ((((((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
                   | (4 * v20 / 3u + 1)
                   | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
                 | (4 * v20 / 3u + 1)
                 | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
                 | (4 * v20 / 3u + 1)
                 | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
               | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
               | (4 * v20 / 3u + 1)
               | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 16;
          v25 = (v24
               | (((((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
                   | (4 * v20 / 3u + 1)
                   | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
                 | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
                 | (4 * v20 / 3u + 1)
                 | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 8)
               | (((((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
                 | (4 * v20 / 3u + 1)
                 | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 4)
               | (((4 * v20 / 3u + 1) | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1)) >> 2)
               | (4 * v20 / 3u + 1)
               | ((unsigned __int64)(4 * v20 / 3u + 1) >> 1))
              + 1;
          *(_DWORD *)(a1 + 1008) = v25;
          v26 = (_QWORD *)sub_C7D670(40 * v25, 8);
          v27 = *(unsigned int *)(a1 + 1008);
          *(_QWORD *)(a1 + 1000) = 0;
          *(_QWORD *)(a1 + 992) = v26;
          for ( i = &v26[5 * v27]; i != v26; v26 += 5 )
          {
            if ( v26 )
              *v26 = -4096;
          }
        }
      }
      else
      {
        if ( (_DWORD)v19 )
        {
          sub_C7D6A0(*(_QWORD *)(a1 + 992), v11, 8);
          v6 = *(_QWORD *)(a1 + 864);
          *(_QWORD *)(a1 + 992) = 0;
          *(_QWORD *)(a1 + 1000) = 0;
          *(_DWORD *)(a1 + 1008) = 0;
          goto LABEL_15;
        }
        *(_QWORD *)(a1 + 1000) = 0;
      }
      v6 = *(_QWORD *)(a1 + 864);
    }
    else
    {
      if ( v12 != v9 )
      {
        do
        {
          if ( *v9 != -4096 )
          {
            if ( *v9 != -8192 )
            {
              v13 = v9[1];
              if ( (_QWORD *)v13 != v9 + 3 )
                _libc_free(v13);
            }
            *v9 = -4096;
          }
          v9 += 5;
        }
        while ( v9 != v12 );
        v6 = *(_QWORD *)(a1 + 864);
      }
      *(_QWORD *)(a1 + 1000) = 0;
    }
  }
LABEL_15:
  v14 = *(_QWORD **)(a1 + 896);
  v15 = sub_2E79000(*(__int64 **)(v6 + 40));
  v16 = *(_QWORD *)(*(_QWORD *)(a1 + 864) + 16LL);
  v14[11] = *(_QWORD *)(a1 + 856);
  v14[12] = v15;
  v14[10] = v16;
  result = sub_AEA460(*(_QWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 864) + 40LL) + 40LL));
  *(_BYTE *)(a1 + 120) = result;
  return result;
}
