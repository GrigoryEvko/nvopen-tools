// Function: sub_A173F0
// Address: 0xa173f0
//
__int64 __fastcall sub_A173F0(__int64 a1, _QWORD *a2)
{
  __int64 v2; // rdi
  _QWORD *v3; // rax
  __int64 v4; // rdx
  _QWORD *v5; // r15
  __int64 v6; // rbx
  __int64 v7; // r12
  volatile signed __int32 *v8; // r13
  signed __int32 v9; // eax
  signed __int32 v10; // eax
  __int64 v11; // rax
  _QWORD *v12; // r15
  __int64 v13; // rbx
  __int64 v14; // r12
  volatile signed __int32 *v15; // r13
  signed __int32 v16; // eax
  signed __int32 v17; // eax
  __int64 v18; // r15
  __int64 v19; // r12
  volatile signed __int32 *v20; // r13
  signed __int32 v21; // eax
  signed __int32 v22; // eax
  __int64 v23; // rax
  __int64 result; // rax
  _QWORD *v25; // r12
  int v26; // r13d
  __int64 v27; // rax
  _QWORD *v28; // [rsp+0h] [rbp-40h]
  _QWORD *v29; // [rsp+0h] [rbp-40h]
  __int64 v30; // [rsp+0h] [rbp-40h]

  if ( *(_DWORD *)(a1 + 48) )
  {
    v25 = *(_QWORD **)(a1 + 24);
    v26 = *(_DWORD *)(a1 + 52);
    v27 = v25[1];
    if ( (unsigned __int64)(v27 + 4) > v25[2] )
    {
      a2 = v25 + 3;
      sub_C8D290(*(_QWORD *)(a1 + 24), v25 + 3, v27 + 4, 1);
      v27 = v25[1];
    }
    *(_DWORD *)(*v25 + v27) = v26;
    v25[1] += 4LL;
    *(_QWORD *)(a1 + 48) = 0;
  }
  v2 = *(_QWORD *)(a1 + 32);
  if ( v2 )
  {
    v3 = *(_QWORD **)(a1 + 24);
    v4 = v3[1];
    if ( v4 )
    {
      a2 = (_QWORD *)*v3;
      sub_CB6200(v2, *v3, v4);
      *(_QWORD *)(*(_QWORD *)(a1 + 24) + 8LL) = 0;
    }
  }
  v5 = *(_QWORD **)(a1 + 128);
  v28 = *(_QWORD **)(a1 + 136);
  if ( v28 != v5 )
  {
    do
    {
      v6 = v5[2];
      v7 = v5[1];
      if ( v6 != v7 )
      {
        do
        {
          while ( 1 )
          {
            v8 = *(volatile signed __int32 **)(v7 + 8);
            if ( v8 )
            {
              if ( &_pthread_key_create )
              {
                v9 = _InterlockedExchangeAdd(v8 + 2, 0xFFFFFFFF);
              }
              else
              {
                v9 = *((_DWORD *)v8 + 2);
                *((_DWORD *)v8 + 2) = v9 - 1;
              }
              if ( v9 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v8 + 16LL))(v8);
                if ( &_pthread_key_create )
                {
                  v10 = _InterlockedExchangeAdd(v8 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v10 = *((_DWORD *)v8 + 3);
                  *((_DWORD *)v8 + 3) = v10 - 1;
                }
                if ( v10 == 1 )
                  break;
              }
            }
            v7 += 16;
            if ( v6 == v7 )
              goto LABEL_17;
          }
          v7 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v8 + 24LL))(v8);
        }
        while ( v6 != v7 );
LABEL_17:
        v7 = v5[1];
      }
      if ( v7 )
      {
        a2 = (_QWORD *)(v5[3] - v7);
        j_j___libc_free_0(v7, a2);
      }
      v5 += 4;
    }
    while ( v28 != v5 );
    v5 = *(_QWORD **)(a1 + 128);
  }
  if ( v5 )
  {
    v11 = *(_QWORD *)(a1 + 144);
    a2 = (_QWORD *)(v11 - (_QWORD)v5);
    j_j___libc_free_0(v5, v11 - (_QWORD)v5);
  }
  v12 = *(_QWORD **)(a1 + 104);
  v29 = *(_QWORD **)(a1 + 112);
  if ( v29 != v12 )
  {
    do
    {
      v13 = v12[3];
      v14 = v12[2];
      if ( v13 != v14 )
      {
        do
        {
          while ( 1 )
          {
            v15 = *(volatile signed __int32 **)(v14 + 8);
            if ( v15 )
            {
              if ( &_pthread_key_create )
              {
                v16 = _InterlockedExchangeAdd(v15 + 2, 0xFFFFFFFF);
              }
              else
              {
                v16 = *((_DWORD *)v15 + 2);
                *((_DWORD *)v15 + 2) = v16 - 1;
              }
              if ( v16 == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 16LL))(v15);
                if ( &_pthread_key_create )
                {
                  v17 = _InterlockedExchangeAdd(v15 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v17 = *((_DWORD *)v15 + 3);
                  *((_DWORD *)v15 + 3) = v17 - 1;
                }
                if ( v17 == 1 )
                  break;
              }
            }
            v14 += 16;
            if ( v13 == v14 )
              goto LABEL_36;
          }
          v14 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 24LL))(v15);
        }
        while ( v13 != v14 );
LABEL_36:
        v14 = v12[2];
      }
      if ( v14 )
      {
        a2 = (_QWORD *)(v12[4] - v14);
        j_j___libc_free_0(v14, a2);
      }
      v12 += 5;
    }
    while ( v29 != v12 );
    v12 = *(_QWORD **)(a1 + 104);
  }
  if ( v12 )
  {
    v30 = *(_QWORD *)(a1 + 120);
    a2 = (_QWORD *)(v30 - (_QWORD)v12);
    j_j___libc_free_0(v12, v30 - (_QWORD)v12);
  }
  v18 = *(_QWORD *)(a1 + 72);
  v19 = *(_QWORD *)(a1 + 64);
  if ( v18 != v19 )
  {
    do
    {
      while ( 1 )
      {
        v20 = *(volatile signed __int32 **)(v19 + 8);
        if ( v20 )
        {
          if ( &_pthread_key_create )
          {
            v21 = _InterlockedExchangeAdd(v20 + 2, 0xFFFFFFFF);
          }
          else
          {
            v21 = *((_DWORD *)v20 + 2);
            *((_DWORD *)v20 + 2) = v21 - 1;
          }
          if ( v21 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v20 + 16LL))(v20);
            if ( &_pthread_key_create )
            {
              v22 = _InterlockedExchangeAdd(v20 + 3, 0xFFFFFFFF);
            }
            else
            {
              v22 = *((_DWORD *)v20 + 3);
              *((_DWORD *)v20 + 3) = v22 - 1;
            }
            if ( v22 == 1 )
              break;
          }
        }
        v19 += 16;
        if ( v18 == v19 )
          goto LABEL_54;
      }
      v19 += 16;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v20 + 24LL))(v20);
    }
    while ( v18 != v19 );
LABEL_54:
    v19 = *(_QWORD *)(a1 + 64);
  }
  if ( v19 )
  {
    v23 = *(_QWORD *)(a1 + 80);
    a2 = (_QWORD *)(v23 - v19);
    j_j___libc_free_0(v19, v23 - v19);
  }
  result = a1;
  if ( *(_QWORD *)a1 != a1 + 24 )
    return _libc_free(*(_QWORD *)a1, a2);
  return result;
}
