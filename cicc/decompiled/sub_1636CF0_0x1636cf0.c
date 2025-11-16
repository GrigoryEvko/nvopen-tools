// Function: sub_1636CF0
// Address: 0x1636cf0
//
__int64 __fastcall sub_1636CF0(__int64 a1)
{
  int v2; // r15d
  __int64 result; // rax
  _QWORD **v4; // rbx
  __int64 v5; // rdx
  _QWORD **v6; // r13
  _QWORD **v7; // r12
  __int64 v8; // rax
  _QWORD *v9; // rbx
  _QWORD *v10; // r15
  __int64 v11; // rdi
  _QWORD **i; // rbx
  __int64 v13; // rax
  _QWORD *v14; // r12
  _QWORD *v15; // r8
  __int64 v16; // rdi
  int v17; // edx
  int v18; // ebx
  unsigned int v19; // r15d
  unsigned int v20; // eax
  _QWORD *v21; // rdi
  unsigned __int64 v22; // rax
  unsigned __int64 v23; // rdi
  __int64 v24; // rdx
  __int64 j; // rdx
  _QWORD *v26; // [rsp+8h] [rbp-38h]

  v2 = *(_DWORD *)(a1 + 16);
  ++*(_QWORD *)a1;
  if ( v2 || (result = *(unsigned int *)(a1 + 20), (_DWORD)result) )
  {
    v4 = *(_QWORD ***)(a1 + 8);
    result = (unsigned int)(4 * v2);
    v5 = *(unsigned int *)(a1 + 24);
    v6 = &v4[4 * v5];
    if ( (unsigned int)result < 0x40 )
      result = 64;
    if ( (unsigned int)v5 <= (unsigned int)result )
    {
      v7 = v4 + 1;
      if ( v4 != v6 )
      {
        while ( 1 )
        {
          v8 = (__int64)*(v7 - 1);
          if ( v8 != -8 )
          {
            if ( v8 != -16 )
            {
              v9 = *v7;
              while ( v7 != v9 )
              {
                v10 = v9;
                v9 = (_QWORD *)*v9;
                v11 = v10[3];
                if ( v11 )
                  (*(void (__fastcall **)(__int64))(*(_QWORD *)v11 + 8LL))(v11);
                j_j___libc_free_0(v10, 32);
              }
            }
            *(v7 - 1) = (_QWORD *)-8LL;
          }
          result = (__int64)(v7 + 4);
          if ( v6 == v7 + 3 )
            break;
          v7 += 4;
        }
      }
LABEL_16:
      *(_QWORD *)(a1 + 16) = 0;
      return result;
    }
    for ( i = v4 + 1; ; i += 4 )
    {
      v13 = (__int64)*(i - 1);
      if ( v13 != -16 && v13 != -8 )
      {
        v14 = *i;
        while ( v14 != i )
        {
          v15 = v14;
          v14 = (_QWORD *)*v14;
          v16 = v15[3];
          if ( v16 )
          {
            v26 = v15;
            (*(void (__fastcall **)(__int64))(*(_QWORD *)v16 + 8LL))(v16);
            v15 = v26;
          }
          j_j___libc_free_0(v15, 32);
        }
      }
      result = (__int64)(i + 4);
      if ( v6 == i + 3 )
        break;
    }
    v17 = *(_DWORD *)(a1 + 24);
    if ( !v2 )
    {
      if ( v17 )
      {
        result = j___libc_free_0(*(_QWORD *)(a1 + 8));
        *(_QWORD *)(a1 + 8) = 0;
        *(_QWORD *)(a1 + 16) = 0;
        *(_DWORD *)(a1 + 24) = 0;
        return result;
      }
      goto LABEL_16;
    }
    v18 = 64;
    v19 = v2 - 1;
    if ( v19 )
    {
      _BitScanReverse(&v20, v19);
      v18 = 1 << (33 - (v20 ^ 0x1F));
      if ( v18 < 64 )
        v18 = 64;
    }
    v21 = *(_QWORD **)(a1 + 8);
    if ( v18 == v17 )
    {
      *(_QWORD *)(a1 + 16) = 0;
      result = (__int64)&v21[4 * (unsigned int)v18];
      do
      {
        if ( v21 )
          *v21 = -8;
        v21 += 4;
      }
      while ( (_QWORD *)result != v21 );
    }
    else
    {
      j___libc_free_0(v21);
      v22 = ((((((((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
               | (4 * v18 / 3u + 1)
               | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 4)
             | (((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
             | (4 * v18 / 3u + 1)
             | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
             | (4 * v18 / 3u + 1)
             | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 4)
           | (((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
           | (4 * v18 / 3u + 1)
           | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 16;
      v23 = (v22
           | (((((((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
               | (4 * v18 / 3u + 1)
               | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 4)
             | (((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
             | (4 * v18 / 3u + 1)
             | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 8)
           | (((((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
             | (4 * v18 / 3u + 1)
             | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 4)
           | (((4 * v18 / 3u + 1) | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1)) >> 2)
           | (4 * v18 / 3u + 1)
           | ((unsigned __int64)(4 * v18 / 3u + 1) >> 1))
          + 1;
      *(_DWORD *)(a1 + 24) = v23;
      result = sub_22077B0(32 * v23);
      v24 = *(unsigned int *)(a1 + 24);
      *(_QWORD *)(a1 + 16) = 0;
      *(_QWORD *)(a1 + 8) = result;
      for ( j = result + 32 * v24; j != result; result += 32 )
      {
        if ( result )
          *(_QWORD *)result = -8;
      }
    }
  }
  return result;
}
