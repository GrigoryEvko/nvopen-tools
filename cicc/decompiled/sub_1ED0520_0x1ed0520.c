// Function: sub_1ED0520
// Address: 0x1ed0520
//
__int64 __fastcall sub_1ED0520(_QWORD *a1)
{
  __int64 v2; // r12
  __int64 result; // rax
  __int64 v4; // rdi
  __int64 v5; // rdi
  __int64 v6; // rdi
  volatile signed __int32 *v7; // rdi
  _QWORD *v8; // [rsp+0h] [rbp-20h] BYREF
  __int64 *v9; // [rsp+8h] [rbp-18h] BYREF

  v2 = a1[4];
  v8 = a1 + 2;
  result = sub_1ED03D0(v2, (unsigned __int64 *)&v8, &v9);
  if ( (_BYTE)result )
  {
    *v9 = 1;
    --*(_DWORD *)(v2 + 16);
    ++*(_DWORD *)(v2 + 20);
  }
  v4 = a1[9];
  if ( v4 )
    result = j_j___libc_free_0_0(v4);
  v5 = a1[8];
  if ( v5 )
    result = j_j___libc_free_0_0(v5);
  v6 = a1[6];
  if ( v6 )
    result = j_j___libc_free_0_0(v6);
  v7 = (volatile signed __int32 *)a1[3];
  if ( v7 )
  {
    if ( &_pthread_key_create )
    {
      result = (unsigned int)_InterlockedExchangeAdd(v7 + 3, 0xFFFFFFFF);
    }
    else
    {
      result = *((unsigned int *)v7 + 3);
      *((_DWORD *)v7 + 3) = result - 1;
    }
    if ( (_DWORD)result == 1 )
      return (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v7 + 24LL))(v7);
  }
  return result;
}
