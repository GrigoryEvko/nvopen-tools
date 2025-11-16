// Function: sub_2DA56F0
// Address: 0x2da56f0
//
_QWORD *__fastcall sub_2DA56F0(_QWORD *a1, _QWORD *a2, unsigned __int64 *a3, unsigned __int64 *a4)
{
  unsigned __int64 i; // r14
  unsigned __int64 v6; // r12
  volatile signed __int32 *v7; // rdi
  signed __int32 v8; // edx
  signed __int32 v9; // edx
  __int64 v13; // [rsp+20h] [rbp-40h] BYREF
  volatile signed __int32 *v14; // [rsp+28h] [rbp-38h]

  for ( i = *a3; (unsigned __int64 *)i != a3; i = *(_QWORD *)i )
  {
    v6 = *a4;
    if ( (unsigned __int64 *)*a4 != a4 )
    {
      while ( 1 )
      {
        if ( !*(_BYTE *)(i + 24) || !*(_BYTE *)(v6 + 24) )
          goto LABEL_4;
        sub_2DA3130(&v13, a2, *(unsigned __int8 **)(i + 16), *(unsigned __int8 **)(v6 + 16));
        if ( v13 )
        {
          --a3[2];
          sub_2208CA0((__int64 *)i);
          j_j___libc_free_0(i);
          --a4[2];
          sub_2208CA0((__int64 *)v6);
          j_j___libc_free_0(v6);
          *a1 = v13;
          a1[1] = v14;
          return a1;
        }
        v7 = v14;
        if ( v14
          && (&_pthread_key_create
            ? (v8 = _InterlockedExchangeAdd(v14 + 2, 0xFFFFFFFF))
            : (v8 = *((_DWORD *)v14 + 2), *((_DWORD *)v14 + 2) = v8 - 1),
              v8 == 1
           && (((*(void (**)(void))(*(_QWORD *)v7 + 16LL))(), &_pthread_key_create)
             ? (v9 = _InterlockedExchangeAdd(v7 + 3, 0xFFFFFFFF))
             : (v9 = *((_DWORD *)v7 + 3), *((_DWORD *)v7 + 3) = v9 - 1),
               v9 == 1)) )
        {
          (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v7 + 24LL))(v7);
          v6 = *(_QWORD *)v6;
          if ( (unsigned __int64 *)v6 == a4 )
            break;
        }
        else
        {
LABEL_4:
          v6 = *(_QWORD *)v6;
          if ( (unsigned __int64 *)v6 == a4 )
            break;
        }
      }
    }
  }
  *a1 = 0;
  a1[1] = 0;
  return a1;
}
