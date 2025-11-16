// Function: sub_2643DC0
// Address: 0x2643dc0
//
_QWORD *__fastcall sub_2643DC0(__int64 a1, _QWORD *a2, _QWORD *a3)
{
  __int64 v3; // r15
  __int64 v4; // r14
  _QWORD *v5; // rbx
  _QWORD *v6; // r12
  __int64 v7; // rsi
  __int64 v8; // rcx
  volatile signed __int32 *v9; // r13
  signed __int32 v10; // ecx
  signed __int32 v11; // ecx
  __int64 v12; // rax
  __int64 v14; // [rsp+0h] [rbp-40h]

  v3 = (__int64)a2 - a1;
  v4 = ((__int64)a2 - a1) >> 4;
  v14 = v4;
  if ( (__int64)a2 - a1 <= 0 )
    return a3;
  v5 = a2;
  v6 = a3;
  do
  {
    while ( 1 )
    {
      v6 -= 2;
      v7 = *(v5 - 2);
      v8 = *(v5 - 1);
      v5 -= 2;
      v5[1] = 0;
      *v5 = 0;
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
      if ( !--v4 )
        goto LABEL_12;
    }
    (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v9 + 24LL))(v9);
    --v4;
  }
  while ( v4 );
LABEL_12:
  v12 = 0x1FFFFFFFFFFFFFFELL;
  if ( v3 > 0 )
    v12 = -2 * v14;
  return &a3[v12];
}
