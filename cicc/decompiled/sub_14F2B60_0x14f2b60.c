// Function: sub_14F2B60
// Address: 0x14f2b60
//
__int64 __fastcall sub_14F2B60(__int64 a1, unsigned __int64 a2)
{
  unsigned __int64 v2; // rbx
  unsigned __int64 v3; // rax
  unsigned __int64 v4; // r12
  __int64 v5; // rax
  unsigned __int64 v6; // rax
  __int64 v7; // rcx
  unsigned __int64 v8; // r15
  __int64 v9; // rdx
  __int64 v10; // rcx
  __int64 v11; // r8
  __int64 v12; // rdi
  __int64 v13; // r12
  __int64 v14; // r14
  volatile signed __int32 *v15; // r13
  signed __int32 v16; // edx
  signed __int32 v17; // edx
  int v19; // [rsp+0h] [rbp-50h]
  __int64 v20; // [rsp+8h] [rbp-48h]
  unsigned __int64 v22; // [rsp+18h] [rbp-38h]

  v2 = a2;
  if ( a2 > 0xFFFFFFFF )
    sub_16BD1C0("SmallVector capacity overflow during allocation");
  v3 = (((((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
         | (*(unsigned int *)(a1 + 12) + 2LL)
         | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
       | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 8)
     | (((((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
       | (*(unsigned int *)(a1 + 12) + 2LL)
       | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 4)
     | (((*(unsigned int *)(a1 + 12) + 2LL) | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1)) >> 2)
     | (*(unsigned int *)(a1 + 12) + 2LL)
     | (((unsigned __int64)*(unsigned int *)(a1 + 12) + 2) >> 1);
  v4 = (v3 | (v3 >> 16) | HIDWORD(v3)) + 1;
  v5 = 0xFFFFFFFFLL;
  if ( v4 >= a2 )
    v2 = v4;
  if ( v2 <= 0xFFFFFFFF )
    v5 = v2;
  v19 = v5;
  v20 = malloc(32 * v5);
  if ( !v20 )
    sub_16BD1C0("Allocation failed");
  v6 = *(_QWORD *)a1;
  v7 = 32LL * *(unsigned int *)(a1 + 8);
  v8 = *(_QWORD *)a1 + v7;
  if ( *(_QWORD *)a1 != v8 )
  {
    v9 = v20;
    v10 = v20 + v7;
    do
    {
      if ( v9 )
      {
        *(_DWORD *)v9 = *(_DWORD *)v6;
        *(_QWORD *)(v9 + 8) = *(_QWORD *)(v6 + 8);
        *(_QWORD *)(v9 + 16) = *(_QWORD *)(v6 + 16);
        *(_QWORD *)(v9 + 24) = *(_QWORD *)(v6 + 24);
        *(_QWORD *)(v6 + 24) = 0;
        *(_QWORD *)(v6 + 16) = 0;
        *(_QWORD *)(v6 + 8) = 0;
      }
      v9 += 32;
      v6 += 32LL;
    }
    while ( v9 != v10 );
    v11 = 32LL * *(unsigned int *)(a1 + 8);
    v22 = *(_QWORD *)a1;
    v8 = *(_QWORD *)a1 + v11;
    if ( v22 != v22 + v11 )
    {
      do
      {
        v12 = *(_QWORD *)(v8 - 24);
        v13 = *(_QWORD *)(v8 - 16);
        v8 -= 32LL;
        v14 = v12;
        if ( v13 != v12 )
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
                goto LABEL_26;
            }
            v14 += 16;
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 24LL))(v15);
          }
          while ( v13 != v14 );
LABEL_26:
          v12 = *(_QWORD *)(v8 + 8);
        }
        if ( v12 )
          j_j___libc_free_0(v12, *(_QWORD *)(v8 + 24) - v12);
      }
      while ( v8 != v22 );
      v8 = *(_QWORD *)a1;
    }
  }
  if ( v8 != a1 + 16 )
    _libc_free(v8);
  *(_QWORD *)a1 = v20;
  *(_DWORD *)(a1 + 12) = v19;
  return a1;
}
