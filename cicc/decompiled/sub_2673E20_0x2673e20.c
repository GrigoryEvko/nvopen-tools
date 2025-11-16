// Function: sub_2673E20
// Address: 0x2673e20
//
__int64 __fastcall sub_2673E20(__int64 a1)
{
  __int64 v1; // r14
  __int64 v2; // r13
  __int64 v3; // rax
  _QWORD *v4; // rbx
  _QWORD *v5; // r15
  volatile signed __int32 *v6; // r12
  signed __int32 v7; // eax
  unsigned __int64 v8; // rdi
  signed __int32 v10; // eax

  v1 = a1 + 34392;
  v2 = a1 + 3352;
  sub_C7D6A0(*(_QWORD *)(a1 + 34952), 8LL * *(unsigned int *)(a1 + 34968), 8);
  sub_C7D6A0(*(_QWORD *)(a1 + 34560), 16LL * *(unsigned int *)(a1 + 34576), 8);
  do
  {
    v3 = *(unsigned int *)(v1 + 152);
    if ( (_DWORD)v3 )
    {
      v4 = *(_QWORD **)(v1 + 136);
      v5 = &v4[3 * v3];
      do
      {
        if ( *v4 != -8192 && *v4 != -4096 )
        {
          v6 = (volatile signed __int32 *)v4[2];
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
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v6 + 16LL))(v6);
              if ( &_pthread_key_create )
              {
                v10 = _InterlockedExchangeAdd(v6 + 3, 0xFFFFFFFF);
              }
              else
              {
                v10 = *((_DWORD *)v6 + 3);
                *((_DWORD *)v6 + 3) = v10 - 1;
              }
              if ( v10 == 1 )
                (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v6 + 24LL))(v6);
            }
          }
        }
        v4 += 3;
      }
      while ( v5 != v4 );
    }
    sub_C7D6A0(*(_QWORD *)(v1 + 136), 24LL * *(unsigned int *)(v1 + 152), 8);
    v8 = *(_QWORD *)(v1 + 40);
    if ( v8 != v1 + 56 )
      _libc_free(v8);
    v1 -= 160;
  }
  while ( v1 != v2 );
  sub_313FB90(a1 + 400);
  return sub_250E960(a1);
}
