// Function: sub_22546A0
// Address: 0x22546a0
//
__int64 __fastcall sub_22546A0(__int64 a1, const char *a2, volatile signed __int32 **a3)
{
  int v4; // r15d
  __int64 v5; // rax
  unsigned __int64 v6; // rbp
  volatile signed __int32 **v7; // r13
  _BYTE *v8; // rsi
  volatile unsigned __int32 v9; // r12d
  volatile signed __int32 *v11[8]; // [rsp+8h] [rbp-40h] BYREF

  if ( &_pthread_key_create && pthread_mutex_lock((pthread_mutex_t *)a1) )
    JUMPOUT(0x426868);
  v4 = *(_DWORD *)(a1 + 40);
  if ( v4 == 0x7FFFFFFF )
  {
    v9 = -1;
  }
  else
  {
    *(_DWORD *)(a1 + 40) = v4 + 1;
    sub_2208E20(v11, a3);
    v5 = sub_22077B0(0x18u);
    *(_DWORD *)v5 = v4;
    v6 = v5;
    v7 = (volatile signed __int32 **)(v5 + 16);
    *(_QWORD *)(v5 + 8) = strdup(a2);
    sub_2208E20(v7, v11);
    sub_2209150(v11);
    if ( *(_QWORD *)(v6 + 8) )
    {
      v11[0] = (volatile signed __int32 *)v6;
      v8 = *(_BYTE **)(a1 + 56);
      if ( v8 == *(_BYTE **)(a1 + 64) )
      {
        sub_2254550(a1 + 48, v8, v11);
      }
      else
      {
        *(_QWORD *)v8 = v6;
        *(_QWORD *)(a1 + 56) = v8 + 8;
      }
      v9 = *(_DWORD *)v6;
    }
    else
    {
      v9 = -1;
      sub_2209150(v7);
      j___libc_free_0(v6);
    }
  }
  if ( &_pthread_key_create && pthread_mutex_unlock((pthread_mutex_t *)a1) )
    JUMPOUT(0x426899);
  return v9;
}
