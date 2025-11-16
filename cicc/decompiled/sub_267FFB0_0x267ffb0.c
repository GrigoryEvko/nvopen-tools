// Function: sub_267FFB0
// Address: 0x267ffb0
//
__int64 __fastcall sub_267FFB0(__int64 *a1)
{
  unsigned int *v1; // r15
  unsigned int v3; // r14d
  __int64 result; // rax
  __int64 *v5; // rbx
  __int64 v6; // rsi
  __int64 *v7; // r13
  volatile signed __int32 *v8; // r14
  volatile signed __int32 *v9; // rdi
  __int64 v10; // rdx
  int v11; // ebx
  unsigned int v12; // eax
  _QWORD *v13; // rdi
  unsigned __int64 v14; // rdx
  unsigned __int64 v15; // rax
  __int64 v16; // rdx
  __int64 i; // rdx
  unsigned int *v18; // [rsp+10h] [rbp-40h]

  v1 = (unsigned int *)(a1 + 439);
  v18 = (unsigned int *)(a1 + 4319);
  do
  {
    v3 = v1[36];
    ++*((_QWORD *)v1 + 16);
    if ( !v3 )
    {
      result = v1[37];
      if ( !(_DWORD)result )
        goto LABEL_18;
    }
    v5 = (__int64 *)*((_QWORD *)v1 + 17);
    result = 4 * v3;
    v6 = 24LL * v1[38];
    if ( (unsigned int)result < 0x40 )
      result = 64;
    v7 = &v5[(unsigned __int64)v6 / 8];
    if ( v1[38] <= (unsigned int)result )
    {
      while ( v5 != v7 )
      {
        result = *v5;
        if ( *v5 != -4096 )
        {
          if ( result != -8192 )
          {
            v8 = (volatile signed __int32 *)v5[2];
            if ( v8 )
            {
              if ( &_pthread_key_create )
              {
                result = (unsigned int)_InterlockedExchangeAdd(v8 + 2, 0xFFFFFFFF);
              }
              else
              {
                result = *((unsigned int *)v8 + 2);
                *((_DWORD *)v8 + 2) = result - 1;
              }
              if ( (_DWORD)result == 1 )
              {
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v8 + 16LL))(v8);
                if ( &_pthread_key_create )
                {
                  result = (unsigned int)_InterlockedExchangeAdd(v8 + 3, 0xFFFFFFFF);
                }
                else
                {
                  result = *((unsigned int *)v8 + 3);
                  *((_DWORD *)v8 + 3) = result - 1;
                }
                if ( (_DWORD)result == 1 )
                  result = (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v8 + 24LL))(v8);
              }
            }
          }
          *v5 = -4096;
        }
        v5 += 3;
      }
LABEL_17:
      v1[36] = 0;
      v1[37] = 0;
      goto LABEL_18;
    }
    do
    {
      result = *v5;
      if ( *v5 != -4096 && result != -8192 )
      {
        v9 = (volatile signed __int32 *)v5[2];
        if ( v9 )
        {
          if ( &_pthread_key_create )
          {
            result = (unsigned int)_InterlockedExchangeAdd(v9 + 2, 0xFFFFFFFF);
          }
          else
          {
            result = *((unsigned int *)v9 + 2);
            *((_DWORD *)v9 + 2) = result - 1;
          }
          if ( (_DWORD)result == 1 )
          {
            (*(void (**)(void))(*(_QWORD *)v9 + 16LL))();
            if ( &_pthread_key_create )
            {
              result = (unsigned int)_InterlockedExchangeAdd(v9 + 3, 0xFFFFFFFF);
            }
            else
            {
              result = *((unsigned int *)v9 + 3);
              *((_DWORD *)v9 + 3) = result - 1;
            }
            if ( (_DWORD)result == 1 )
              result = (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v9 + 24LL))(v9);
          }
        }
      }
      v5 += 3;
    }
    while ( v5 != v7 );
    v10 = v1[38];
    if ( v3 )
    {
      v11 = 64;
      if ( v3 != 1 )
      {
        _BitScanReverse(&v12, v3 - 1);
        v11 = 1 << (33 - (v12 ^ 0x1F));
        if ( v11 < 64 )
          v11 = 64;
      }
      v13 = (_QWORD *)*((_QWORD *)v1 + 17);
      if ( (_DWORD)v10 == v11 )
      {
        v1[36] = 0;
        v1[37] = 0;
        result = (__int64)&v13[3 * v10];
        do
        {
          if ( v13 )
            *v13 = -4096;
          v13 += 3;
        }
        while ( (_QWORD *)result != v13 );
      }
      else
      {
        sub_C7D6A0((__int64)v13, v6, 8);
        v14 = ((((((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
                 | (4 * v11 / 3u + 1)
                 | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
               | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
               | (4 * v11 / 3u + 1)
               | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 8)
             | (((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
               | (4 * v11 / 3u + 1)
               | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
             | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
             | (4 * v11 / 3u + 1)
             | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 16;
        v15 = (v14
             | (((((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
                 | (4 * v11 / 3u + 1)
                 | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
               | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
               | (4 * v11 / 3u + 1)
               | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 8)
             | (((((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
               | (4 * v11 / 3u + 1)
               | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 4)
             | (((4 * v11 / 3u + 1) | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1)) >> 2)
             | (4 * v11 / 3u + 1)
             | ((unsigned __int64)(4 * v11 / 3u + 1) >> 1))
            + 1;
        v1[38] = v15;
        result = sub_C7D670(24 * v15, 8);
        v16 = v1[38];
        v1[36] = 0;
        *((_QWORD *)v1 + 17) = result;
        v1[37] = 0;
        for ( i = result + 24 * v16; i != result; result += 24 )
        {
          if ( result )
            *(_QWORD *)result = -4096;
        }
      }
    }
    else
    {
      if ( !(_DWORD)v10 )
        goto LABEL_17;
      result = sub_C7D6A0(*((_QWORD *)v1 + 17), v6, 8);
      v1[38] = 0;
      *((_QWORD *)v1 + 17) = 0;
      v1[36] = 0;
      v1[37] = 0;
    }
LABEL_18:
    if ( *((_QWORD *)v1 + 15) )
    {
      sub_3122A50(a1 + 50, *v1);
      result = sub_267FDF0(a1, (__int64)v1);
    }
    v1 += 40;
  }
  while ( v18 != v1 );
  return result;
}
