// Function: sub_1516EE0
// Address: 0x1516ee0
//
void __fastcall sub_1516EE0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  __int64 v4; // rdi
  __int64 v5; // r8
  unsigned __int64 v6; // r14
  __int64 v7; // rdi
  __int64 v8; // rdx
  __int64 v9; // rbx
  volatile signed __int32 *v10; // r12
  signed __int32 v11; // eax
  signed __int32 v12; // eax
  __int64 v13; // rbx
  __int64 v14; // r12
  volatile signed __int32 *v15; // r14
  signed __int32 v16; // eax
  signed __int32 v17; // eax
  void (__fastcall *v18)(__int64, __int64, __int64); // rax
  _QWORD *v19; // rbx
  _QWORD *v20; // r12
  __int64 v21; // rdi
  _QWORD *v22; // r12
  _QWORD *v23; // rbx
  unsigned __int64 v24; // rbx
  unsigned __int64 v25; // r12
  __int64 v26; // rsi
  __int64 v27; // rax
  __int64 v28; // [rsp+0h] [rbp-40h]
  __int64 v29; // [rsp+8h] [rbp-38h]

  j___libc_free_0(*(_QWORD *)(a1 + 984));
  if ( (*(_BYTE *)(a1 + 712) & 1) == 0 )
    j___libc_free_0(*(_QWORD *)(a1 + 720));
  v2 = *(_QWORD *)(a1 + 680);
  if ( v2 )
    j_j___libc_free_0(v2, *(_QWORD *)(a1 + 696) - v2);
  v3 = *(_QWORD *)(a1 + 656);
  if ( v3 )
    j_j___libc_free_0(v3, *(_QWORD *)(a1 + 672) - v3);
  v4 = *(_QWORD *)(a1 + 632);
  if ( v4 )
    j_j___libc_free_0(v4, *(_QWORD *)(a1 + 648) - v4);
  v5 = 32LL * *(unsigned int *)(a1 + 360);
  v28 = *(_QWORD *)(a1 + 352);
  v6 = v28 + v5;
  if ( v28 != v28 + v5 )
  {
    do
    {
      v7 = *(_QWORD *)(v6 - 24);
      v8 = *(_QWORD *)(v6 - 16);
      v6 -= 32LL;
      v9 = v7;
      if ( v8 != v7 )
      {
        do
        {
          while ( 1 )
          {
            v10 = *(volatile signed __int32 **)(v9 + 8);
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
                v29 = v8;
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v10 + 16LL))(v10);
                v8 = v29;
                if ( &_pthread_key_create )
                {
                  v12 = _InterlockedExchangeAdd(v10 + 3, 0xFFFFFFFF);
                }
                else
                {
                  v12 = *((_DWORD *)v10 + 3);
                  *((_DWORD *)v10 + 3) = v12 - 1;
                }
                if ( v12 == 1 )
                  break;
              }
            }
            v9 += 16;
            if ( v8 == v9 )
              goto LABEL_21;
          }
          v9 += 16;
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v10 + 24LL))(v10);
          v8 = v29;
        }
        while ( v29 != v9 );
LABEL_21:
        v7 = *(_QWORD *)(v6 + 8);
      }
      if ( v7 )
        j_j___libc_free_0(v7, *(_QWORD *)(v6 + 24) - v7);
    }
    while ( v28 != v6 );
    v6 = *(_QWORD *)(a1 + 352);
  }
  if ( v6 != a1 + 368 )
    _libc_free(v6);
  v13 = *(_QWORD *)(a1 + 336);
  v14 = *(_QWORD *)(a1 + 328);
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
          goto LABEL_39;
      }
      v14 += 16;
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v15 + 24LL))(v15);
    }
    while ( v13 != v14 );
LABEL_39:
    v14 = *(_QWORD *)(a1 + 328);
  }
  if ( v14 )
    j_j___libc_free_0(v14, *(_QWORD *)(a1 + 344) - v14);
  v18 = *(void (__fastcall **)(__int64, __int64, __int64))(a1 + 272);
  if ( v18 )
    v18(a1 + 256, a1 + 256, 3);
  v19 = *(_QWORD **)(a1 + 184);
  v20 = &v19[2 * *(unsigned int *)(a1 + 192)];
  if ( v19 != v20 )
  {
    do
    {
      v21 = *(v20 - 1);
      v20 -= 2;
      if ( v21 )
        sub_16307F0();
      if ( *v20 )
        sub_161E7C0(v20);
    }
    while ( v19 != v20 );
    v20 = *(_QWORD **)(a1 + 184);
  }
  if ( v20 != (_QWORD *)(a1 + 200) )
    _libc_free((unsigned __int64)v20);
  if ( (*(_BYTE *)(a1 + 160) & 1) == 0 )
    j___libc_free_0(*(_QWORD *)(a1 + 168));
  if ( (*(_BYTE *)(a1 + 128) & 1) == 0 )
    j___libc_free_0(*(_QWORD *)(a1 + 136));
  if ( (*(_BYTE *)(a1 + 96) & 1) != 0 )
  {
    v22 = (_QWORD *)(a1 + 104);
    v23 = (_QWORD *)(a1 + 120);
  }
  else
  {
    v27 = *(unsigned int *)(a1 + 112);
    v22 = *(_QWORD **)(a1 + 104);
    if ( !(_DWORD)v27 )
    {
LABEL_82:
      j___libc_free_0(v22);
      goto LABEL_65;
    }
    v23 = &v22[2 * v27];
  }
  do
  {
    if ( *v22 != -16 && *v22 != -8 && v22[1] )
      sub_16307F0();
    v22 += 2;
  }
  while ( v23 != v22 );
  if ( (*(_BYTE *)(a1 + 96) & 1) == 0 )
  {
    v22 = *(_QWORD **)(a1 + 104);
    goto LABEL_82;
  }
LABEL_65:
  if ( (*(_BYTE *)(a1 + 64) & 1) == 0 )
    j___libc_free_0(*(_QWORD *)(a1 + 72));
  if ( (*(_BYTE *)(a1 + 32) & 1) == 0 )
    j___libc_free_0(*(_QWORD *)(a1 + 40));
  v24 = *(_QWORD *)a1;
  v25 = *(_QWORD *)a1 + 8LL * *(unsigned int *)(a1 + 8);
  if ( *(_QWORD *)a1 != v25 )
  {
    do
    {
      v26 = *(_QWORD *)(v25 - 8);
      v25 -= 8LL;
      if ( v26 )
        sub_161E7C0(v25);
    }
    while ( v24 != v25 );
    v25 = *(_QWORD *)a1;
  }
  if ( v25 != a1 + 16 )
    _libc_free(v25);
}
