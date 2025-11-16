// Function: sub_1C07C70
// Address: 0x1c07c70
//
__int64 __fastcall sub_1C07C70(__int64 a1)
{
  __int64 v2; // rax
  __int64 v3; // r14
  __int64 v4; // rbx
  __int64 v5; // r13
  unsigned __int64 v6; // rdi
  _QWORD *v8; // rax
  _QWORD *v9; // r13
  _QWORD *v10; // rbx
  __int64 v11; // r14
  unsigned __int64 v12; // rdi
  unsigned __int64 v13; // rdi
  _DWORD *v14; // rax
  _DWORD *v15; // r13
  _DWORD *v16; // rbx
  __int64 v17; // rdi
  _QWORD *v18; // rax
  _QWORD *v19; // r13
  _QWORD *v20; // rbx
  unsigned __int64 *v21; // r14

  if ( *(_DWORD *)(a1 + 56) )
  {
    v8 = *(_QWORD **)(a1 + 48);
    v9 = &v8[2 * *(unsigned int *)(a1 + 64)];
    if ( v8 != v9 )
    {
      while ( 1 )
      {
        v10 = v8;
        if ( *v8 != -8 && *v8 != -16 )
          break;
        v8 += 2;
        if ( v9 == v8 )
          goto LABEL_2;
      }
      while ( v9 != v10 )
      {
        v11 = v10[1];
        if ( v11 )
        {
          v12 = *(_QWORD *)(v11 + 48);
          if ( v12 != v11 + 64 )
            _libc_free(v12);
          v13 = *(_QWORD *)(v11 + 24);
          if ( v13 != v11 + 40 )
            _libc_free(v13);
          j_j___libc_free_0(v11, 80);
        }
        v10 += 2;
        if ( v10 == v9 )
          break;
        while ( *v10 == -16 || *v10 == -8 )
        {
          v10 += 2;
          if ( v9 == v10 )
            goto LABEL_2;
        }
      }
    }
  }
LABEL_2:
  v2 = *(unsigned int *)(a1 + 112);
  if ( (_DWORD)v2 )
  {
    v3 = 8 * v2;
    v4 = 0;
    do
    {
      v5 = *(_QWORD *)(*(_QWORD *)(a1 + 104) + v4);
      if ( v5 )
      {
        j___libc_free_0(*(_QWORD *)(v5 + 8));
        j_j___libc_free_0(v5, 32);
      }
      v4 += 8;
    }
    while ( v3 != v4 );
  }
  if ( *(_DWORD *)(a1 + 24) )
  {
    v18 = *(_QWORD **)(a1 + 16);
    v19 = &v18[2 * *(unsigned int *)(a1 + 32)];
    if ( v18 != v19 )
    {
      while ( 1 )
      {
        v20 = v18;
        if ( *v18 != -16 && *v18 != -8 )
          break;
        v18 += 2;
        if ( v19 == v18 )
          goto LABEL_8;
      }
      if ( v19 != v18 )
      {
        do
        {
          v21 = (unsigned __int64 *)v20[1];
          if ( v21 )
          {
            _libc_free(*v21);
            j_j___libc_free_0(v21, 24);
          }
          v20 += 2;
          if ( v20 == v19 )
            break;
          while ( *v20 == -8 || *v20 == -16 )
          {
            v20 += 2;
            if ( v19 == v20 )
              goto LABEL_8;
          }
        }
        while ( v20 != v19 );
      }
    }
  }
LABEL_8:
  if ( *(_DWORD *)(a1 + 88) )
  {
    v14 = *(_DWORD **)(a1 + 80);
    v15 = &v14[4 * *(unsigned int *)(a1 + 96)];
    if ( v14 != v15 )
    {
      while ( 1 )
      {
        v16 = v14;
        if ( (unsigned int)(*v14 + 0x7FFFFFFF) <= 0xFFFFFFFD )
          break;
        v14 += 4;
        if ( v15 == v14 )
          goto LABEL_9;
      }
      while ( v16 != v15 )
      {
        v17 = *((_QWORD *)v16 + 1);
        if ( v17 )
          j_j___libc_free_0(v17, 24);
        v16 += 4;
        if ( v16 == v15 )
          break;
        while ( (unsigned int)(*v16 + 0x7FFFFFFF) > 0xFFFFFFFD )
        {
          v16 += 4;
          if ( v15 == v16 )
            goto LABEL_9;
        }
      }
    }
  }
LABEL_9:
  v6 = *(_QWORD *)(a1 + 104);
  if ( v6 != a1 + 120 )
    _libc_free(v6);
  j___libc_free_0(*(_QWORD *)(a1 + 80));
  j___libc_free_0(*(_QWORD *)(a1 + 48));
  return j___libc_free_0(*(_QWORD *)(a1 + 16));
}
