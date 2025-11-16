// Function: sub_2643F00
// Address: 0x2643f00
//
char *__fastcall sub_2643F00(__int64 *a1, __int64 a2, _QWORD *a3)
{
  __int64 v3; // rsi
  __int64 v4; // r15
  __int64 *v5; // rbx
  _QWORD *v6; // r12
  __int64 v7; // rcx
  __int64 v8; // rdx
  volatile signed __int32 *v9; // r13
  signed __int32 v10; // edx
  signed __int32 v11; // edx

  v3 = a2 - (_QWORD)a1;
  v4 = v3 >> 4;
  if ( v3 <= 0 )
    return (char *)a3;
  v5 = a1;
  v6 = a3;
  do
  {
    while ( 1 )
    {
      v7 = *v5;
      v8 = v5[1];
      *v5 = 0;
      v5[1] = 0;
      v9 = (volatile signed __int32 *)v6[1];
      *v6 = v7;
      v6[1] = v8;
      if ( v9 )
      {
        if ( &_pthread_key_create )
        {
          v10 = _InterlockedExchangeAdd(v9 + 2, 0xFFFFFFFF);
        }
        else
        {
          v10 = *((_DWORD *)v9 + 2);
          *((_DWORD *)v9 + 2) = v10 - 1;
        }
        if ( v10 == 1 )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v9 + 16LL))(v9);
          if ( &_pthread_key_create )
          {
            v11 = _InterlockedExchangeAdd(v9 + 3, 0xFFFFFFFF);
          }
          else
          {
            v11 = *((_DWORD *)v9 + 3);
            *((_DWORD *)v9 + 3) = v11 - 1;
          }
          if ( v11 == 1 )
            break;
        }
      }
      v5 += 2;
      v6 += 2;
      if ( !--v4 )
        return (char *)a3 + v3;
    }
    v5 += 2;
    v6 += 2;
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v9 + 24LL))(v9);
    --v4;
  }
  while ( v4 );
  return (char *)a3 + v3;
}
