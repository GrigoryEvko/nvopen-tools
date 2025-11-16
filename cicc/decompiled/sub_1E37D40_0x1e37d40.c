// Function: sub_1E37D40
// Address: 0x1e37d40
//
char *__fastcall sub_1E37D40(char *a1, char *a2, char *a3, char *a4, _QWORD *a5)
{
  char *v8; // rbx
  __int64 v9; // rax
  volatile signed __int32 *v10; // r14
  signed __int32 v11; // eax
  _DWORD *v12; // rdx
  _DWORD *v13; // rcx
  __int64 v14; // rax
  volatile signed __int32 *v15; // r14
  signed __int32 v16; // eax
  __int64 v17; // rcx
  __int64 v18; // r15
  _QWORD *v19; // rbx
  _DWORD *v20; // rsi
  __int64 v21; // rdx
  volatile signed __int32 *v22; // r14
  signed __int32 v23; // esi
  signed __int32 v24; // edx
  signed __int32 v26; // eax
  signed __int32 v27; // eax
  __int64 v28; // r15
  _QWORD *v29; // r12
  _DWORD *v30; // rcx
  __int64 v31; // rdx
  volatile signed __int32 *v32; // r14
  signed __int32 v33; // ecx
  signed __int32 v34; // edx
  __int64 v35; // rax
  __int64 v36; // [rsp+0h] [rbp-40h]
  __int64 v37; // [rsp+8h] [rbp-38h]

  if ( a1 == a2 )
  {
LABEL_15:
    v17 = a4 - a3;
    v18 = (a4 - a3) >> 4;
    if ( v17 > 0 )
    {
      v19 = a5;
      do
      {
        while ( 1 )
        {
          v20 = *(_DWORD **)a3;
          v21 = *((_QWORD *)a3 + 1);
          *(_QWORD *)a3 = 0;
          *((_QWORD *)a3 + 1) = 0;
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
              v37 = v17;
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v22 + 16LL))(v22);
              v17 = v37;
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
          a3 += 16;
          v19 += 2;
          if ( !--v18 )
            return (char *)a5 + v17;
        }
        a3 += 16;
        v19 += 2;
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v22 + 24LL))(v22);
        v17 = v37;
        --v18;
      }
      while ( v18 );
      return (char *)a5 + v17;
    }
    return (char *)a5;
  }
  v8 = a1;
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
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v10 + 16LL))(v10);
          if ( &_pthread_key_create )
          {
            v26 = _InterlockedExchangeAdd(v10 + 3, 0xFFFFFFFF);
          }
          else
          {
            v26 = *((_DWORD *)v10 + 3);
            *((_DWORD *)v10 + 3) = v26 - 1;
          }
          if ( v26 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v10 + 24LL))(v10);
        }
      }
      a3 += 16;
      a5 += 2;
      if ( v8 == a2 )
        goto LABEL_15;
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
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 16LL))(v15);
          if ( &_pthread_key_create )
          {
            v27 = _InterlockedExchangeAdd(v15 + 3, 0xFFFFFFFF);
          }
          else
          {
            v27 = *((_DWORD *)v15 + 3);
            *((_DWORD *)v15 + 3) = v27 - 1;
          }
          if ( v27 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 24LL))(v15);
        }
      }
      v8 += 16;
      a5 += 2;
      if ( v8 == a2 )
        goto LABEL_15;
    }
  }
  v36 = a2 - v8;
  v28 = (a2 - v8) >> 4;
  if ( a2 - v8 <= 0 )
    return (char *)a5;
  v29 = a5;
  do
  {
    while ( 1 )
    {
      v30 = *(_DWORD **)v8;
      v31 = *((_QWORD *)v8 + 1);
      *(_QWORD *)v8 = 0;
      *((_QWORD *)v8 + 1) = 0;
      v32 = (volatile signed __int32 *)v29[1];
      *v29 = v30;
      v29[1] = v31;
      if ( v32 )
      {
        if ( &_pthread_key_create )
        {
          v33 = _InterlockedExchangeAdd(v32 + 2, 0xFFFFFFFF);
        }
        else
        {
          v33 = *((_DWORD *)v32 + 2);
          *((_DWORD *)v32 + 2) = v33 - 1;
        }
        if ( v33 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v32 + 16LL))(v32);
          if ( &_pthread_key_create )
          {
            v34 = _InterlockedExchangeAdd(v32 + 3, 0xFFFFFFFF);
          }
          else
          {
            v34 = *((_DWORD *)v32 + 3);
            *((_DWORD *)v32 + 3) = v34 - 1;
          }
          if ( v34 == 1 )
            break;
        }
      }
      v8 += 16;
      v29 += 2;
      if ( !--v28 )
        goto LABEL_49;
    }
    v8 += 16;
    v29 += 2;
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v32 + 24LL))(v32);
    --v28;
  }
  while ( v28 );
LABEL_49:
  v35 = 16;
  if ( v36 > 0 )
    v35 = v36;
  return (char *)a5 + v35;
}
