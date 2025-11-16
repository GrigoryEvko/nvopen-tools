// Function: sub_14EB5C0
// Address: 0x14eb5c0
//
__int64 __fastcall sub_14EB5C0(__int64 a1)
{
  unsigned int v1; // ecx
  __int64 v2; // r15
  __int64 v3; // rax
  __int64 v4; // r13
  __int64 v5; // rdx
  volatile signed __int32 *v6; // r12
  signed __int32 v7; // eax
  signed __int32 v8; // eax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r15
  __int64 v12; // r12
  __int64 v13; // r14
  volatile signed __int32 *v14; // r13
  signed __int32 v15; // eax
  signed __int32 v16; // eax
  __int64 v18; // [rsp+0h] [rbp-40h]
  __int64 v19; // [rsp+8h] [rbp-38h]

  v1 = *(_DWORD *)(a1 + 32);
  if ( v1 > 0x1F )
  {
    *(_DWORD *)(a1 + 32) = 32;
    *(_QWORD *)(a1 + 24) >>= (unsigned __int8)v1 - 32;
  }
  else
  {
    *(_DWORD *)(a1 + 32) = 0;
  }
  v2 = *(_QWORD *)(a1 + 40);
  v3 = *(_QWORD *)(a1 + 64) + 32LL * *(unsigned int *)(a1 + 72) - 32;
  v18 = *(_QWORD *)(a1 + 56);
  v4 = v2;
  *(_DWORD *)(a1 + 36) = *(_DWORD *)v3;
  v5 = *(_QWORD *)(a1 + 48);
  *(_QWORD *)(a1 + 40) = *(_QWORD *)(v3 + 8);
  *(_QWORD *)(a1 + 48) = *(_QWORD *)(v3 + 16);
  *(_QWORD *)(a1 + 56) = *(_QWORD *)(v3 + 24);
  *(_QWORD *)(v3 + 8) = 0;
  *(_QWORD *)(v3 + 16) = 0;
  *(_QWORD *)(v3 + 24) = 0;
  if ( v2 != v5 )
  {
    do
    {
      while ( 1 )
      {
        v6 = *(volatile signed __int32 **)(v4 + 8);
        if ( v6 )
        {
          if ( &_pthread_key_create )
          {
            v7 = _InterlockedExchangeAdd(v6 + 2, 0xFFFFFFFF);
          }
          else
          {
            v7 = *((_DWORD *)v6 + 2);
            *((_DWORD *)v6 + 2) = v7 - 1;
          }
          if ( v7 == 1 )
          {
            v19 = v5;
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v6 + 16LL))(v6);
            v5 = v19;
            if ( &_pthread_key_create )
            {
              v8 = _InterlockedExchangeAdd(v6 + 3, 0xFFFFFFFF);
            }
            else
            {
              v8 = *((_DWORD *)v6 + 3);
              *((_DWORD *)v6 + 3) = v8 - 1;
            }
            if ( v8 == 1 )
              break;
          }
        }
        v4 += 16;
        if ( v5 == v4 )
          goto LABEL_14;
      }
      v4 += 16;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v6 + 24LL))(v6);
      v5 = v19;
    }
    while ( v19 != v4 );
  }
LABEL_14:
  if ( v2 )
    j_j___libc_free_0(v2, v18 - v2);
  v9 = (unsigned int)(*(_DWORD *)(a1 + 72) - 1);
  *(_DWORD *)(a1 + 72) = v9;
  v10 = *(_QWORD *)(a1 + 64) + 32 * v9;
  v11 = *(_QWORD *)(v10 + 16);
  v12 = *(_QWORD *)(v10 + 8);
  v13 = v10;
  if ( v11 != v12 )
  {
    do
    {
      while ( 1 )
      {
        v14 = *(volatile signed __int32 **)(v12 + 8);
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
        v12 += 16;
        if ( v11 == v12 )
          goto LABEL_27;
      }
      v12 += 16;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v14 + 24LL))(v14);
    }
    while ( v11 != v12 );
LABEL_27:
    v12 = *(_QWORD *)(v13 + 8);
  }
  if ( v12 )
    j_j___libc_free_0(v12, *(_QWORD *)(v13 + 24) - v12);
  return 0;
}
