// Function: sub_EFDF60
// Address: 0xefdf60
//
__int64 __fastcall sub_EFDF60(__int64 a1, unsigned __int8 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rdi
  __int64 v7; // rax
  size_t v8; // rdx
  _QWORD *v9; // r15
  __int64 v10; // rbx
  __int64 v11; // r12
  volatile signed __int32 *v12; // r13
  signed __int32 v13; // eax
  signed __int32 v14; // eax
  __int64 v15; // rax
  _QWORD *v16; // r15
  __int64 v17; // rbx
  __int64 v18; // r12
  volatile signed __int32 *v19; // r13
  signed __int32 v20; // eax
  signed __int32 v21; // eax
  __int64 v22; // r15
  __int64 v23; // r12
  volatile signed __int32 *v24; // r13
  signed __int32 v25; // eax
  signed __int32 v26; // eax
  __int64 v27; // rax
  __int64 v28; // rdi
  __int64 v29; // rdi
  __int64 result; // rax
  _QWORD *v31; // r12
  int v32; // r13d
  __int64 v33; // rax
  _QWORD *v34; // [rsp+0h] [rbp-40h]
  _QWORD *v35; // [rsp+0h] [rbp-40h]
  __int64 v36; // [rsp+0h] [rbp-40h]

  if ( *(_DWORD *)(a1 + 1624) )
  {
    v31 = *(_QWORD **)(a1 + 1600);
    v32 = *(_DWORD *)(a1 + 1628);
    v33 = v31[1];
    if ( (unsigned __int64)(v33 + 4) > v31[2] )
    {
      a2 = (unsigned __int8 *)(v31 + 3);
      sub_C8D290(*(_QWORD *)(a1 + 1600), v31 + 3, v33 + 4, 1u, a5, a6);
      v33 = v31[1];
    }
    *(_DWORD *)(*v31 + v33) = v32;
    v31[1] += 4LL;
    *(_QWORD *)(a1 + 1624) = 0;
  }
  v6 = *(_QWORD *)(a1 + 1608);
  if ( v6 )
  {
    v7 = *(_QWORD *)(a1 + 1600);
    v8 = *(_QWORD *)(v7 + 8);
    if ( v8 )
    {
      a2 = *(unsigned __int8 **)v7;
      sub_CB6200(v6, *(unsigned __int8 **)v7, v8);
      *(_QWORD *)(*(_QWORD *)(a1 + 1600) + 8LL) = 0;
    }
  }
  v9 = *(_QWORD **)(a1 + 1704);
  v34 = *(_QWORD **)(a1 + 1712);
  if ( v34 != v9 )
  {
    do
    {
      v10 = v9[2];
      v11 = v9[1];
      if ( v10 != v11 )
      {
        do
        {
          while ( 1 )
          {
            v12 = *(volatile signed __int32 **)(v11 + 8);
            if ( v12 )
            {
              if ( &_pthread_key_create )
              {
                v13 = _InterlockedExchangeAdd(v12 + 2, 0xFFFFFFFF);
              }
              else
              {
                v13 = *((_DWORD *)v12 + 2);
                *((_DWORD *)v12 + 2) = v13 - 1;
              }
              if ( v13 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v12 + 16LL))(v12);
                if ( &_pthread_key_create )
                {
                  v14 = _InterlockedExchangeAdd(v12 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v14 = *((_DWORD *)v12 + 3);
                  *((_DWORD *)v12 + 3) = v14 - 1;
                }
                if ( v14 == 1 )
                  break;
              }
            }
            v11 += 16;
            if ( v10 == v11 )
              goto LABEL_17;
          }
          v11 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v12 + 24LL))(v12);
        }
        while ( v10 != v11 );
LABEL_17:
        v11 = v9[1];
      }
      if ( v11 )
      {
        a2 = (unsigned __int8 *)(v9[3] - v11);
        j_j___libc_free_0(v11, a2);
      }
      v9 += 4;
    }
    while ( v34 != v9 );
    v9 = *(_QWORD **)(a1 + 1704);
  }
  if ( v9 )
  {
    v15 = *(_QWORD *)(a1 + 1720);
    a2 = (unsigned __int8 *)(v15 - (_QWORD)v9);
    j_j___libc_free_0(v9, v15 - (_QWORD)v9);
  }
  v16 = *(_QWORD **)(a1 + 1680);
  v35 = *(_QWORD **)(a1 + 1688);
  if ( v35 != v16 )
  {
    do
    {
      v17 = v16[3];
      v18 = v16[2];
      if ( v17 != v18 )
      {
        do
        {
          while ( 1 )
          {
            v19 = *(volatile signed __int32 **)(v18 + 8);
            if ( v19 )
            {
              if ( &_pthread_key_create )
              {
                v20 = _InterlockedExchangeAdd(v19 + 2, 0xFFFFFFFF);
              }
              else
              {
                v20 = *((_DWORD *)v19 + 2);
                *((_DWORD *)v19 + 2) = v20 - 1;
              }
              if ( v20 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v19 + 16LL))(v19);
                if ( &_pthread_key_create )
                {
                  v21 = _InterlockedExchangeAdd(v19 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v21 = *((_DWORD *)v19 + 3);
                  *((_DWORD *)v19 + 3) = v21 - 1;
                }
                if ( v21 == 1 )
                  break;
              }
            }
            v18 += 16;
            if ( v17 == v18 )
              goto LABEL_36;
          }
          v18 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v19 + 24LL))(v19);
        }
        while ( v17 != v18 );
LABEL_36:
        v18 = v16[2];
      }
      if ( v18 )
      {
        a2 = (unsigned __int8 *)(v16[4] - v18);
        j_j___libc_free_0(v18, a2);
      }
      v16 += 5;
    }
    while ( v35 != v16 );
    v16 = *(_QWORD **)(a1 + 1680);
  }
  if ( v16 )
  {
    v36 = *(_QWORD *)(a1 + 1696);
    a2 = (unsigned __int8 *)(v36 - (_QWORD)v16);
    j_j___libc_free_0(v16, v36 - (_QWORD)v16);
  }
  v22 = *(_QWORD *)(a1 + 1648);
  v23 = *(_QWORD *)(a1 + 1640);
  if ( v22 != v23 )
  {
    do
    {
      while ( 1 )
      {
        v24 = *(volatile signed __int32 **)(v23 + 8);
        if ( v24 )
        {
          if ( &_pthread_key_create )
          {
            v25 = _InterlockedExchangeAdd(v24 + 2, 0xFFFFFFFF);
          }
          else
          {
            v25 = *((_DWORD *)v24 + 2);
            *((_DWORD *)v24 + 2) = v25 - 1;
          }
          if ( v25 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v24 + 16LL))(v24);
            if ( &_pthread_key_create )
            {
              v26 = _InterlockedExchangeAdd(v24 + 3, 0xFFFFFFFF);
            }
            else
            {
              v26 = *((_DWORD *)v24 + 3);
              *((_DWORD *)v24 + 3) = v26 - 1;
            }
            if ( v26 == 1 )
              break;
          }
        }
        v23 += 16;
        if ( v22 == v23 )
          goto LABEL_54;
      }
      v23 += 16;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v24 + 24LL))(v24);
    }
    while ( v22 != v23 );
LABEL_54:
    v23 = *(_QWORD *)(a1 + 1640);
  }
  if ( v23 )
  {
    v27 = *(_QWORD *)(a1 + 1656);
    a2 = (unsigned __int8 *)(v27 - v23);
    j_j___libc_free_0(v23, v27 - v23);
  }
  v28 = *(_QWORD *)(a1 + 1576);
  if ( v28 != a1 + 1600 )
    _libc_free(v28, a2);
  v29 = *(_QWORD *)(a1 + 1048);
  if ( v29 != a1 + 1064 )
    _libc_free(v29, a2);
  result = a1;
  if ( *(_QWORD *)a1 != a1 + 24 )
    return _libc_free(*(_QWORD *)a1, a2);
  return result;
}
