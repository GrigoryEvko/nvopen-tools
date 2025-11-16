// Function: sub_2261E30
// Address: 0x2261e30
//
__int64 __fastcall sub_2261E30(unsigned __int64 *a1, unsigned __int64 *a2, int a3)
{
  unsigned __int64 v4; // r12
  volatile signed __int32 *v5; // r13
  signed __int32 v6; // eax
  signed __int32 v7; // eax
  _QWORD *v8; // r12
  _QWORD *v9; // rax
  __int64 v10; // rdx

  if ( a3 == 1 )
  {
    *a1 = *a2;
    return 0;
  }
  if ( a3 != 2 )
  {
    if ( a3 == 3 )
    {
      v4 = *a1;
      if ( *a1 )
      {
        v5 = *(volatile signed __int32 **)(v4 + 8);
        if ( v5 )
        {
          if ( &_pthread_key_create )
          {
            v6 = _InterlockedExchangeAdd(v5 + 2, 0xFFFFFFFF);
          }
          else
          {
            v6 = *((_DWORD *)v5 + 2);
            *((_DWORD *)v5 + 2) = v6 - 1;
          }
          if ( v6 == 1 )
          {
            (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v5 + 16LL))(v5);
            if ( &_pthread_key_create )
            {
              v7 = _InterlockedExchangeAdd(v5 + 3, 0xFFFFFFFF);
            }
            else
            {
              v7 = *((_DWORD *)v5 + 3);
              *((_DWORD *)v5 + 3) = v7 - 1;
            }
            if ( v7 == 1 )
              (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v5 + 24LL))(v5);
          }
        }
        j_j___libc_free_0(v4);
      }
    }
    return 0;
  }
  v8 = (_QWORD *)*a2;
  v9 = (_QWORD *)sub_22077B0(0x10u);
  if ( v9 )
  {
    *v9 = *v8;
    v10 = v8[1];
    v9[1] = v10;
    if ( v10 )
    {
      if ( &_pthread_key_create )
        _InterlockedAdd((volatile signed __int32 *)(v10 + 8), 1u);
      else
        ++*(_DWORD *)(v10 + 8);
    }
  }
  *a1 = (unsigned __int64)v9;
  return 0;
}
