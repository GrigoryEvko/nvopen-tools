// Function: sub_1E38140
// Address: 0x1e38140
//
_QWORD *__fastcall sub_1E38140(char *a1, char *a2, char *a3, char *a4, _QWORD *a5)
{
  char *v8; // rbx
  __int64 v9; // rax
  volatile signed __int32 *v10; // r15
  signed __int32 v11; // edx
  _DWORD *v12; // rax
  _DWORD *v13; // rdx
  __int64 v14; // rdx
  volatile signed __int32 *v15; // r15
  signed __int32 v16; // edx
  __int64 v17; // rsi
  __int64 v18; // rdx
  _QWORD *v19; // r14
  _DWORD *v20; // rdi
  __int64 v21; // rax
  volatile signed __int32 *v22; // r15
  signed __int32 v23; // edi
  signed __int32 v24; // eax
  __int64 v25; // r14
  __int64 v26; // rcx
  __int64 v27; // r15
  _QWORD *v28; // rbx
  _DWORD *v29; // rdx
  __int64 v30; // rax
  volatile signed __int32 *v31; // r14
  signed __int32 v32; // edx
  signed __int32 v33; // eax
  signed __int32 v35; // eax
  signed __int32 v36; // eax
  char *v37; // [rsp+8h] [rbp-48h]
  __int64 v38; // [rsp+18h] [rbp-38h]
  __int64 v39; // [rsp+18h] [rbp-38h]
  char *v40; // [rsp+18h] [rbp-38h]
  char *v41; // [rsp+18h] [rbp-38h]

  v8 = a1;
  if ( a2 != a1 )
  {
    while ( a4 != a3 )
    {
      v12 = *(_DWORD **)v8;
      v13 = *(_DWORD **)a3;
      if ( **(_DWORD **)a3 > **(_DWORD **)v8 )
      {
        *(_QWORD *)a3 = 0;
        v9 = *((_QWORD *)a3 + 1);
        *((_QWORD *)a3 + 1) = 0;
        v10 = (volatile signed __int32 *)a5[1];
        *a5 = v13;
        a5[1] = v9;
        if ( v10 )
        {
          if ( &_pthread_key_create )
          {
            v11 = _InterlockedExchangeAdd(v10 + 2, 0xFFFFFFFF);
          }
          else
          {
            v11 = *((_DWORD *)v10 + 2);
            *((_DWORD *)v10 + 2) = v11 - 1;
          }
          if ( v11 == 1 )
          {
            v40 = a4;
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v10 + 16LL))(v10);
            a4 = v40;
            if ( &_pthread_key_create )
            {
              v35 = _InterlockedExchangeAdd(v10 + 3, 0xFFFFFFFF);
            }
            else
            {
              v35 = *((_DWORD *)v10 + 3);
              *((_DWORD *)v10 + 3) = v35 - 1;
            }
            if ( v35 == 1 )
            {
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v10 + 24LL))(v10);
              a4 = v40;
            }
          }
        }
        a3 += 16;
        a5 += 2;
        if ( a2 == v8 )
          break;
      }
      else
      {
        *(_QWORD *)v8 = 0;
        v14 = *((_QWORD *)v8 + 1);
        *((_QWORD *)v8 + 1) = 0;
        v15 = (volatile signed __int32 *)a5[1];
        *a5 = v12;
        a5[1] = v14;
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
            v41 = a4;
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 16LL))(v15);
            a4 = v41;
            if ( &_pthread_key_create )
            {
              v36 = _InterlockedExchangeAdd(v15 + 3, 0xFFFFFFFF);
            }
            else
            {
              v36 = *((_DWORD *)v15 + 3);
              *((_DWORD *)v15 + 3) = v36 - 1;
            }
            if ( v36 == 1 )
            {
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 24LL))(v15);
              a4 = v41;
            }
          }
        }
        v8 += 16;
        a5 += 2;
        if ( a2 == v8 )
          break;
      }
    }
  }
  v17 = a2 - v8;
  v18 = (a2 - v8) >> 4;
  if ( a2 - v8 > 0 )
  {
    v19 = a5;
    do
    {
      while ( 1 )
      {
        v20 = *(_DWORD **)v8;
        v21 = *((_QWORD *)v8 + 1);
        *(_QWORD *)v8 = 0;
        *((_QWORD *)v8 + 1) = 0;
        v22 = (volatile signed __int32 *)v19[1];
        *v19 = v20;
        v19[1] = v21;
        if ( v22 )
        {
          if ( &_pthread_key_create )
          {
            v23 = _InterlockedExchangeAdd(v22 + 2, 0xFFFFFFFF);
          }
          else
          {
            v23 = *((_DWORD *)v22 + 2);
            *((_DWORD *)v22 + 2) = v23 - 1;
          }
          if ( v23 == 1 )
          {
            v37 = a4;
            v38 = v18;
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v22 + 16LL))(v22);
            v18 = v38;
            a4 = v37;
            if ( &_pthread_key_create )
            {
              v24 = _InterlockedExchangeAdd(v22 + 3, 0xFFFFFFFF);
            }
            else
            {
              v24 = *((_DWORD *)v22 + 3);
              *((_DWORD *)v22 + 3) = v24 - 1;
            }
            if ( v24 == 1 )
              break;
          }
        }
        v8 += 16;
        v19 += 2;
        if ( !--v18 )
          goto LABEL_26;
      }
      v8 += 16;
      v19 += 2;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v22 + 24LL))(v22);
      a4 = v37;
      v18 = v38 - 1;
    }
    while ( v38 != 1 );
LABEL_26:
    v25 = 16;
    if ( v17 > 0 )
      v25 = v17;
    a5 = (_QWORD *)((char *)a5 + v25);
  }
  v26 = a4 - a3;
  v27 = v26 >> 4;
  if ( v26 > 0 )
  {
    v28 = a5;
    do
    {
      while ( 1 )
      {
        v29 = *(_DWORD **)a3;
        v30 = *((_QWORD *)a3 + 1);
        *(_QWORD *)a3 = 0;
        *((_QWORD *)a3 + 1) = 0;
        v31 = (volatile signed __int32 *)v28[1];
        *v28 = v29;
        v28[1] = v30;
        if ( v31 )
        {
          if ( &_pthread_key_create )
          {
            v32 = _InterlockedExchangeAdd(v31 + 2, 0xFFFFFFFF);
          }
          else
          {
            v32 = *((_DWORD *)v31 + 2);
            *((_DWORD *)v31 + 2) = v32 - 1;
          }
          if ( v32 == 1 )
          {
            v39 = v26;
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v31 + 16LL))(v31);
            v26 = v39;
            if ( &_pthread_key_create )
            {
              v33 = _InterlockedExchangeAdd(v31 + 3, 0xFFFFFFFF);
            }
            else
            {
              v33 = *((_DWORD *)v31 + 3);
              *((_DWORD *)v31 + 3) = v33 - 1;
            }
            if ( v33 == 1 )
              break;
          }
        }
        a3 += 16;
        v28 += 2;
        if ( !--v27 )
          return (_QWORD *)((char *)a5 + v26);
      }
      a3 += 16;
      v28 += 2;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v31 + 24LL))(v31);
      v26 = v39;
      --v27;
    }
    while ( v27 );
    return (_QWORD *)((char *)a5 + v26);
  }
  return a5;
}
