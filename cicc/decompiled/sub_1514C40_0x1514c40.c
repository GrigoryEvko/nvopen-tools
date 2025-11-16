// Function: sub_1514C40
// Address: 0x1514c40
//
void __fastcall sub_1514C40(_QWORD *a1, _QWORD *a2)
{
  _QWORD *v2; // rdx
  char *v3; // r15
  char *v4; // rbx
  char *v5; // r12
  __int64 v6; // r14
  char *v7; // r13
  __int64 v8; // rax
  _QWORD *v9; // r15
  __int64 v10; // rsi
  volatile signed __int32 *v11; // rdi
  volatile signed __int32 *v12; // rcx
  signed __int32 v13; // r8d
  signed __int32 v14; // eax
  signed __int32 v15; // eax
  volatile signed __int32 *v16; // r15
  signed __int32 v17; // ecx
  __int64 v18; // rax
  _QWORD *v19; // r12
  _QWORD *v20; // rax
  _QWORD *v21; // r15
  __int64 v22; // rcx
  char *v23; // rbx
  char *v24; // r13
  volatile signed __int32 *v25; // r15
  signed __int32 v26; // ecx
  signed __int32 v27; // eax
  char *v28; // r14
  char *v29; // rbx
  char *v30; // r15
  __int64 v31; // rax
  __int64 v32; // rcx
  volatile signed __int32 *v33; // r13
  volatile signed __int32 *v34; // r15
  signed __int32 v35; // edi
  signed __int32 v36; // eax
  _QWORD *v37; // [rsp-58h] [rbp-58h]
  volatile signed __int32 *v38; // [rsp-50h] [rbp-50h]
  _QWORD *v39; // [rsp-48h] [rbp-48h]
  _QWORD *v40; // [rsp-40h] [rbp-40h]
  _QWORD *v41; // [rsp-40h] [rbp-40h]
  _QWORD *v42; // [rsp-40h] [rbp-40h]
  __int64 v43; // [rsp-40h] [rbp-40h]

  if ( a2 != a1 )
  {
    v2 = a1;
    v3 = (char *)a2[1];
    v4 = (char *)*a2;
    v5 = (char *)*a1;
    v6 = (__int64)&v3[-*a2];
    if ( a1[2] - *a1 < (unsigned __int64)v6 )
    {
      if ( v6 )
      {
        if ( (unsigned __int64)v6 > 0x7FFFFFFFFFFFFFF0LL )
          sub_4261EA(a1, a2, a1);
        v18 = sub_22077B0(v6);
        v2 = a1;
        v19 = (_QWORD *)v18;
      }
      else
      {
        v19 = 0;
      }
      if ( v3 != v4 )
      {
        v20 = v19;
        v21 = (_QWORD *)((char *)v19 + v3 - v4);
        do
        {
          if ( v20 )
          {
            *v20 = *(_QWORD *)v4;
            v22 = *((_QWORD *)v4 + 1);
            v20[1] = v22;
            if ( v22 )
            {
              if ( &_pthread_key_create )
                _InterlockedAdd((volatile signed __int32 *)(v22 + 8), 1u);
              else
                ++*(_DWORD *)(v22 + 8);
            }
          }
          v20 += 2;
          v4 += 16;
        }
        while ( v20 != v21 );
      }
      v23 = (char *)v2[1];
      v24 = (char *)*v2;
      if ( v23 != (char *)*v2 )
      {
        do
        {
          while ( 1 )
          {
            v25 = (volatile signed __int32 *)*((_QWORD *)v24 + 1);
            if ( v25 )
            {
              if ( &_pthread_key_create )
              {
                v26 = _InterlockedExchangeAdd(v25 + 2, 0xFFFFFFFF);
              }
              else
              {
                v26 = *((_DWORD *)v25 + 2);
                *((_DWORD *)v25 + 2) = v26 - 1;
              }
              if ( v26 == 1 )
              {
                v41 = v2;
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v25 + 16LL))(v25);
                v2 = v41;
                if ( &_pthread_key_create )
                {
                  v27 = _InterlockedExchangeAdd(v25 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v27 = *((_DWORD *)v25 + 3);
                  *((_DWORD *)v25 + 3) = v27 - 1;
                }
                if ( v27 == 1 )
                  break;
              }
            }
            v24 += 16;
            if ( v23 == v24 )
              goto LABEL_53;
          }
          v24 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v25 + 24LL))(v25);
          v2 = v41;
        }
        while ( v23 != v24 );
LABEL_53:
        v24 = (char *)*v2;
      }
      if ( v24 )
      {
        v42 = v2;
        j_j___libc_free_0(v24, v2[2] - (_QWORD)v24);
        v2 = v42;
      }
      v28 = (char *)v19 + v6;
      *v2 = v19;
      v2[2] = v28;
      goto LABEL_66;
    }
    v7 = (char *)a1[1];
    v8 = v7 - v5;
    if ( v6 > (unsigned __int64)(v7 - v5) )
    {
      v32 = v8 >> 4;
      if ( v8 > 0 )
      {
        do
        {
          v33 = (volatile signed __int32 *)*((_QWORD *)v5 + 1);
          *(_QWORD *)v5 = *(_QWORD *)v4;
          v34 = (volatile signed __int32 *)*((_QWORD *)v4 + 1);
          if ( v34 != v33 )
          {
            if ( v34 )
            {
              if ( &_pthread_key_create )
                _InterlockedAdd(v34 + 2, 1u);
              else
                ++*((_DWORD *)v34 + 2);
              v33 = (volatile signed __int32 *)*((_QWORD *)v5 + 1);
            }
            if ( v33 )
            {
              if ( &_pthread_key_create )
              {
                v35 = _InterlockedExchangeAdd(v33 + 2, 0xFFFFFFFF);
              }
              else
              {
                v35 = *((_DWORD *)v33 + 2);
                *((_DWORD *)v33 + 2) = v35 - 1;
              }
              if ( v35 == 1 )
              {
                v39 = v2;
                v43 = v32;
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v33 + 16LL))(v33);
                v32 = v43;
                v2 = v39;
                if ( &_pthread_key_create )
                {
                  v36 = _InterlockedExchangeAdd(v33 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v36 = *((_DWORD *)v33 + 3);
                  *((_DWORD *)v33 + 3) = v36 - 1;
                }
                if ( v36 == 1 )
                {
                  (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v33 + 24LL))(v33);
                  v2 = v39;
                  v32 = v43;
                }
              }
            }
            *((_QWORD *)v5 + 1) = v34;
          }
          v4 += 16;
          v5 += 16;
          --v32;
        }
        while ( v32 );
        v7 = (char *)v2[1];
        v5 = (char *)*v2;
        v3 = (char *)a2[1];
        v4 = (char *)*a2;
        v8 = (__int64)&v7[-*v2];
      }
      v29 = &v4[v8];
      if ( v29 == v3 )
      {
        v28 = &v5[v6];
        goto LABEL_66;
      }
      v30 = &v7[v3 - v29];
      do
      {
        if ( v7 )
        {
          *(_QWORD *)v7 = *(_QWORD *)v29;
          v31 = *((_QWORD *)v29 + 1);
          *((_QWORD *)v7 + 1) = v31;
          if ( v31 )
          {
            if ( &_pthread_key_create )
              _InterlockedAdd((volatile signed __int32 *)(v31 + 8), 1u);
            else
              ++*(_DWORD *)(v31 + 8);
          }
        }
        v7 += 16;
        v29 += 16;
      }
      while ( v7 != v30 );
    }
    else
    {
      v9 = (_QWORD *)*a1;
      v10 = v6 >> 4;
      if ( v6 > 0 )
      {
        do
        {
          v11 = (volatile signed __int32 *)v9[1];
          *v9 = *(_QWORD *)v4;
          v12 = (volatile signed __int32 *)*((_QWORD *)v4 + 1);
          if ( v12 != v11 )
          {
            if ( v12 )
            {
              if ( &_pthread_key_create )
                _InterlockedAdd(v12 + 2, 1u);
              else
                ++*((_DWORD *)v12 + 2);
              v11 = (volatile signed __int32 *)v9[1];
            }
            if ( v11 )
            {
              if ( &_pthread_key_create )
              {
                v13 = _InterlockedExchangeAdd(v11 + 2, 0xFFFFFFFF);
              }
              else
              {
                v13 = *((_DWORD *)v11 + 2);
                *((_DWORD *)v11 + 2) = v13 - 1;
              }
              if ( v13 == 1 )
              {
                v37 = v2;
                v38 = v12;
                (*(void (**)(void))(*(_QWORD *)v11 + 16LL))();
                v12 = v38;
                v2 = v37;
                if ( &_pthread_key_create )
                {
                  v14 = _InterlockedExchangeAdd(v11 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v14 = *((_DWORD *)v11 + 3);
                  *((_DWORD *)v11 + 3) = v14 - 1;
                }
                if ( v14 == 1 )
                {
                  (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v11 + 24LL))(v11);
                  v2 = v37;
                  v12 = v38;
                }
              }
            }
            v9[1] = v12;
          }
          v4 += 16;
          v9 += 2;
          --v10;
        }
        while ( v10 );
        v5 += v6;
      }
      while ( v7 != v5 )
      {
        v16 = (volatile signed __int32 *)*((_QWORD *)v5 + 1);
        if ( v16 )
        {
          if ( &_pthread_key_create )
          {
            v17 = _InterlockedExchangeAdd(v16 + 2, 0xFFFFFFFF);
          }
          else
          {
            v17 = *((_DWORD *)v16 + 2);
            *((_DWORD *)v16 + 2) = v17 - 1;
          }
          if ( v17 == 1 )
          {
            v40 = v2;
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v16 + 16LL))(v16);
            v2 = v40;
            if ( &_pthread_key_create )
            {
              v15 = _InterlockedExchangeAdd(v16 + 3, 0xFFFFFFFF);
            }
            else
            {
              v15 = *((_DWORD *)v16 + 3);
              *((_DWORD *)v16 + 3) = v15 - 1;
            }
            if ( v15 == 1 )
            {
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v16 + 24LL))(v16);
              v2 = v40;
            }
          }
        }
        v5 += 16;
      }
    }
    v28 = (char *)(*v2 + v6);
LABEL_66:
    v2[1] = v28;
  }
}
