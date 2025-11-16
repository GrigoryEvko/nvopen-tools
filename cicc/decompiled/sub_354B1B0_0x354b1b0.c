// Function: sub_354B1B0
// Address: 0x354b1b0
//
void __fastcall sub_354B1B0(__int64 a1, unsigned __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v7; // rax
  _QWORD *v8; // rdx
  _QWORD *v9; // r13
  __int64 v10; // rcx
  unsigned __int64 v11; // r12
  _QWORD *v12; // rcx
  _QWORD *v13; // r15
  unsigned __int64 v14; // r14
  volatile signed __int32 *v15; // rdi
  signed __int32 v16; // edx
  volatile signed __int32 *v17; // rdi
  signed __int32 v18; // edx
  int v19; // r15d
  signed __int32 v20; // eax
  signed __int32 v21; // eax
  __int64 v22; // [rsp+18h] [rbp-48h]
  unsigned __int64 v23[7]; // [rsp+28h] [rbp-38h] BYREF

  v22 = a1 + 16;
  v7 = (_QWORD *)sub_C8D7D0(a1, a1 + 16, a2, 8u, v23, a6);
  v8 = *(_QWORD **)a1;
  v9 = v7;
  v10 = *(unsigned int *)(a1 + 8);
  v11 = *(_QWORD *)a1 + v10 * 8;
  if ( *(_QWORD *)a1 != v11 )
  {
    v12 = &v7[v10];
    do
    {
      if ( v7 )
      {
        *v7 = *v8;
        *v8 = 0;
      }
      ++v7;
      ++v8;
    }
    while ( v7 != v12 );
    v13 = *(_QWORD **)a1;
    v11 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
    if ( *(_QWORD *)a1 != v11 )
    {
      do
      {
        v14 = *(_QWORD *)(v11 - 8);
        v11 -= 8LL;
        if ( v14 )
        {
          v15 = *(volatile signed __int32 **)(v14 + 32);
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
              (*(void (**)(void))(*(_QWORD *)v15 + 16LL))();
              if ( &_pthread_key_create )
              {
                v21 = _InterlockedExchangeAdd(v15 + 3, 0xFFFFFFFF);
              }
              else
              {
                v21 = *((_DWORD *)v15 + 3);
                *((_DWORD *)v15 + 3) = v21 - 1;
              }
              if ( v21 == 1 )
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 24LL))(v15);
            }
          }
          v17 = *(volatile signed __int32 **)(v14 + 16);
          if ( v17 )
          {
            if ( &_pthread_key_create )
            {
              v18 = _InterlockedExchangeAdd(v17 + 2, 0xFFFFFFFF);
            }
            else
            {
              v18 = *((_DWORD *)v17 + 2);
              *((_DWORD *)v17 + 2) = v18 - 1;
            }
            if ( v18 == 1 )
            {
              (*(void (**)(void))(*(_QWORD *)v17 + 16LL))();
              if ( &_pthread_key_create )
              {
                v20 = _InterlockedExchangeAdd(v17 + 3, 0xFFFFFFFF);
              }
              else
              {
                v20 = *((_DWORD *)v17 + 3);
                *((_DWORD *)v17 + 3) = v20 - 1;
              }
              if ( v20 == 1 )
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v17 + 24LL))(v17);
            }
          }
          j_j___libc_free_0(v14);
        }
      }
      while ( v13 != (_QWORD *)v11 );
      v11 = *(_QWORD *)a1;
    }
  }
  v19 = v23[0];
  if ( v22 != v11 )
    _libc_free(v11);
  *(_QWORD *)a1 = v9;
  *(_DWORD *)(a1 + 12) = v19;
}
