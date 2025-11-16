// Function: sub_1E38550
// Address: 0x1e38550
//
void __fastcall sub_1E38550(__int64 a1, unsigned __int64 *a2)
{
  unsigned __int64 *v2; // r15
  unsigned __int64 v4; // rsi
  unsigned __int64 *v5; // rbx
  unsigned __int64 v6; // rax
  bool v7; // cc
  __int64 v8; // rax
  unsigned __int64 v9; // rdi
  unsigned __int64 v10; // rdx
  volatile signed __int32 *v11; // r12
  signed __int32 v12; // edi
  signed __int32 v13; // edx
  volatile signed __int32 *v14; // r12
  signed __int32 v15; // eax
  _DWORD *v16; // rax
  unsigned __int64 *v17; // r12
  unsigned __int64 *v18; // rdx
  unsigned __int64 v19; // rdx
  volatile signed __int32 *v20; // rbx
  signed __int32 v21; // eax
  signed __int32 v22; // eax
  volatile signed __int32 *v23; // rbx
  signed __int32 v24; // eax
  signed __int32 v25; // eax
  signed __int32 v26; // eax
  __int64 v28; // [rsp+8h] [rbp-48h]
  unsigned __int64 v29; // [rsp+18h] [rbp-38h]

  if ( (unsigned __int64 *)a1 != a2 )
  {
    v2 = (unsigned __int64 *)(a1 + 16);
    while ( a2 != v2 )
    {
      v4 = *v2;
      v5 = v2;
      v6 = v2[1];
      v2 += 2;
      v29 = v6;
      v7 = *(_DWORD *)v4 <= **(_DWORD **)a1;
      *(v2 - 1) = 0;
      *(v2 - 2) = 0;
      if ( v7 )
      {
        v16 = (_DWORD *)*(v2 - 4);
        v17 = v2 - 4;
        if ( *(_DWORD *)v4 <= *v16 )
        {
          *v5 = v4;
          v5[1] = v29;
          continue;
        }
        while ( 1 )
        {
          v19 = v17[1];
          v20 = (volatile signed __int32 *)v17[3];
          v17[2] = (unsigned __int64)v16;
          v17[1] = 0;
          *v17 = 0;
          v17[3] = v19;
          if ( v20
            && (&_pthread_key_create
              ? (v21 = _InterlockedExchangeAdd(v20 + 2, 0xFFFFFFFF))
              : (v21 = *((_DWORD *)v20 + 2), *((_DWORD *)v20 + 2) = v21 - 1),
                v21 == 1
             && (((*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v20 + 16LL))(v20), &_pthread_key_create)
               ? (v22 = _InterlockedExchangeAdd(v20 + 3, 0xFFFFFFFF))
               : (v22 = *((_DWORD *)v20 + 3), *((_DWORD *)v20 + 3) = v22 - 1),
                 v22 == 1)) )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v20 + 24LL))(v20);
            v16 = (_DWORD *)*(v17 - 2);
            v18 = v17 - 2;
            if ( *(_DWORD *)v4 <= *v16 )
            {
LABEL_34:
              v23 = (volatile signed __int32 *)v17[1];
              *v17 = v4;
              v17[1] = v29;
              if ( v23 )
              {
                if ( &_pthread_key_create )
                {
                  v24 = _InterlockedExchangeAdd(v23 + 2, 0xFFFFFFFF);
                }
                else
                {
                  v24 = *((_DWORD *)v23 + 2);
                  v4 = (unsigned int)(v24 - 1);
                  *((_DWORD *)v23 + 2) = v4;
                }
                if ( v24 == 1 )
                {
                  (*(void (__fastcall **)(volatile signed __int32 *, unsigned __int64, unsigned __int64 *))(*(_QWORD *)v23 + 16LL))(
                    v23,
                    v4,
                    v18);
                  if ( &_pthread_key_create )
                  {
                    v25 = _InterlockedExchangeAdd(v23 + 3, 0xFFFFFFFF);
                  }
                  else
                  {
                    v25 = *((_DWORD *)v23 + 3);
                    *((_DWORD *)v23 + 3) = v25 - 1;
                  }
                  if ( v25 == 1 )
                    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v23 + 24LL))(v23);
                }
              }
              goto LABEL_19;
            }
          }
          else
          {
            v16 = (_DWORD *)*(v17 - 2);
            v18 = v17 - 2;
            if ( *(_DWORD *)v4 <= *v16 )
              goto LABEL_34;
          }
          v17 = v18;
        }
      }
      v8 = ((__int64)v5 - a1) >> 4;
      if ( (__int64)v5 - a1 > 0 )
      {
        do
        {
          while ( 1 )
          {
            v9 = *(v5 - 2);
            v10 = *(v5 - 1);
            v5 -= 2;
            v5[1] = 0;
            v11 = (volatile signed __int32 *)v5[3];
            *v5 = 0;
            v5[3] = v10;
            v5[2] = v9;
            if ( v11 )
            {
              if ( &_pthread_key_create )
              {
                v12 = _InterlockedExchangeAdd(v11 + 2, 0xFFFFFFFF);
              }
              else
              {
                v12 = *((_DWORD *)v11 + 2);
                *((_DWORD *)v11 + 2) = v12 - 1;
              }
              if ( v12 == 1 )
              {
                v28 = v8;
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v11 + 16LL))(v11);
                v8 = v28;
                if ( &_pthread_key_create )
                {
                  v13 = _InterlockedExchangeAdd(v11 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v13 = *((_DWORD *)v11 + 3);
                  *((_DWORD *)v11 + 3) = v13 - 1;
                }
                if ( v13 == 1 )
                  break;
              }
            }
            if ( !--v8 )
              goto LABEL_15;
          }
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v11 + 24LL))(v11);
          v8 = v28 - 1;
        }
        while ( v28 != 1 );
      }
LABEL_15:
      v14 = *(volatile signed __int32 **)(a1 + 8);
      *(_QWORD *)a1 = v4;
      *(_QWORD *)(a1 + 8) = v29;
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
            v26 = _InterlockedExchangeAdd(v14 + 3, 0xFFFFFFFF);
          }
          else
          {
            v26 = *((_DWORD *)v14 + 3);
            *((_DWORD *)v14 + 3) = v26 - 1;
          }
          if ( v26 == 1 )
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v14 + 24LL))(v14);
        }
      }
LABEL_19:
      ;
    }
  }
}
