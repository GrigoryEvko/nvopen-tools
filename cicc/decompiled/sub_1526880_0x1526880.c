// Function: sub_1526880
// Address: 0x1526880
//
__int64 __fastcall sub_1526880(__int64 a1)
{
  __int64 v2; // rdi
  unsigned __int64 *v3; // rbx
  unsigned __int64 *v4; // r12
  unsigned __int64 v5; // rdi
  unsigned __int64 *v6; // rbx
  unsigned __int64 v7; // r12
  unsigned __int64 v8; // rdi
  unsigned __int64 v9; // rdi
  __int64 result; // rax
  _QWORD *v11; // r15
  __int64 v12; // r14
  __int64 v13; // r12
  volatile signed __int32 *v14; // r13
  signed __int32 v15; // eax
  signed __int32 v16; // eax
  _QWORD *v17; // r15
  __int64 v18; // r14
  __int64 v19; // r12
  volatile signed __int32 *v20; // r13
  signed __int32 v21; // eax
  signed __int32 v22; // eax
  __int64 v23; // rbx
  __int64 v24; // r12
  volatile signed __int32 *v25; // rdi
  _QWORD *v26; // [rsp+0h] [rbp-40h]
  _QWORD *v27; // [rsp+8h] [rbp-38h]
  _QWORD *v28; // [rsp+8h] [rbp-38h]

  v2 = *(_QWORD *)(a1 + 184);
  if ( v2 )
    j_j___libc_free_0(v2, *(_QWORD *)(a1 + 200) - v2);
  v3 = *(unsigned __int64 **)(a1 + 88);
  v4 = &v3[*(unsigned int *)(a1 + 96)];
  while ( v4 != v3 )
  {
    v5 = *v3++;
    _libc_free(v5);
  }
  v6 = *(unsigned __int64 **)(a1 + 136);
  v7 = (unsigned __int64)&v6[2 * *(unsigned int *)(a1 + 144)];
  if ( v6 != (unsigned __int64 *)v7 )
  {
    do
    {
      v8 = *v6;
      v6 += 2;
      _libc_free(v8);
    }
    while ( (unsigned __int64 *)v7 != v6 );
    v7 = *(_QWORD *)(a1 + 136);
  }
  if ( v7 != a1 + 152 )
    _libc_free(v7);
  v9 = *(_QWORD *)(a1 + 88);
  if ( v9 != a1 + 104 )
    _libc_free(v9);
  sub_167FA50(a1 + 16);
  result = *(_QWORD *)(a1 + 8);
  v26 = (_QWORD *)result;
  if ( result )
  {
    v11 = *(_QWORD **)(result + 72);
    v27 = *(_QWORD **)(result + 80);
    if ( v27 != v11 )
    {
      do
      {
        v12 = v11[2];
        v13 = v11[1];
        if ( v12 != v13 )
        {
          do
          {
            while ( 1 )
            {
              v14 = *(volatile signed __int32 **)(v13 + 8);
              if ( v14 )
              {
                if ( &_pthread_key_create )
                {
                  v15 = _InterlockedExchangeAdd(v14 + 2, 0xFFFFFFFF);
                }
                else
                {
                  v15 = *((_DWORD *)v14 + 2);
                  *((_DWORD *)v14 + 2) = v15 - 1;
                }
                if ( v15 == 1 )
                {
                  (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v14 + 16LL))(v14);
                  if ( &_pthread_key_create )
                  {
                    v16 = _InterlockedExchangeAdd(v14 + 3, 0xFFFFFFFF);
                  }
                  else
                  {
                    v16 = *((_DWORD *)v14 + 3);
                    *((_DWORD *)v14 + 3) = v16 - 1;
                  }
                  if ( v16 == 1 )
                    break;
                }
              }
              v13 += 16;
              if ( v12 == v13 )
                goto LABEL_25;
            }
            v13 += 16;
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v14 + 24LL))(v14);
          }
          while ( v12 != v13 );
LABEL_25:
          v13 = v11[1];
        }
        if ( v13 )
          j_j___libc_free_0(v13, v11[3] - v13);
        v11 += 4;
      }
      while ( v27 != v11 );
      v11 = (_QWORD *)v26[9];
    }
    if ( v11 )
      j_j___libc_free_0(v11, v26[11] - (_QWORD)v11);
    v17 = (_QWORD *)v26[6];
    v28 = (_QWORD *)v26[7];
    if ( v28 != v17 )
    {
      do
      {
        v18 = v17[3];
        v19 = v17[2];
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
                goto LABEL_44;
            }
            v19 += 16;
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v20 + 24LL))(v20);
          }
          while ( v18 != v19 );
LABEL_44:
          v19 = v17[2];
        }
        if ( v19 )
          j_j___libc_free_0(v19, v17[4] - v19);
        v17 += 5;
      }
      while ( v28 != v17 );
      v17 = (_QWORD *)v26[6];
    }
    if ( v17 )
      j_j___libc_free_0(v17, v26[8] - (_QWORD)v17);
    v23 = v26[4];
    v24 = v26[3];
    if ( v23 != v24 )
    {
      do
      {
        v25 = *(volatile signed __int32 **)(v24 + 8);
        if ( v25 )
          sub_A191D0(v25);
        v24 += 16;
      }
      while ( v23 != v24 );
      v24 = v26[3];
    }
    if ( v24 )
      j_j___libc_free_0(v24, v26[5] - v24);
    return j_j___libc_free_0(v26, 96);
  }
  return result;
}
