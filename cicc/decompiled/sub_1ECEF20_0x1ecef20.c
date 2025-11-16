// Function: sub_1ECEF20
// Address: 0x1ecef20
//
__int64 __fastcall sub_1ECEF20(_QWORD *a1)
{
  __int64 v2; // r12
  __int64 result; // rax
  __int64 v4; // rdi
  volatile signed __int32 *v5; // rdi
  _QWORD *v6; // [rsp+0h] [rbp-20h] BYREF
  __int64 *v7; // [rsp+8h] [rbp-18h] BYREF

  v2 = a1[4];
  v6 = a1 + 2;
  result = sub_1ECEDE0(v2, (unsigned __int64 *)&v6, &v7);
  if ( (_BYTE)result )
  {
    *v7 = 1;
    --*(_DWORD *)(v2 + 16);
    ++*(_DWORD *)(v2 + 20);
  }
  v4 = a1[6];
  if ( v4 )
    result = j_j___libc_free_0_0(v4);
  v5 = (volatile signed __int32 *)a1[3];
  if ( v5 )
  {
    if ( &_pthread_key_create )
    {
      result = (unsigned int)_InterlockedExchangeAdd(v5 + 3, 0xFFFFFFFF);
    }
    else
    {
      result = *((unsigned int *)v5 + 3);
      *((_DWORD *)v5 + 3) = result - 1;
    }
    if ( (_DWORD)result == 1 )
      return (*(__int64 (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v5 + 24LL))(v5);
  }
  return result;
}
