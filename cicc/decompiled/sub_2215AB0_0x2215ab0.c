// Function: sub_2215AB0
// Address: 0x2215ab0
//
void *__fastcall sub_2215AB0(__int64 *a1, unsigned __int64 a2)
{
  __int64 v3; // rdi
  unsigned __int64 v4; // rax
  unsigned __int64 v5; // rdx
  void *result; // rax
  __int64 v7; // rdx
  void *v8; // rdi
  int v9; // ecx
  void *v10; // [rsp+8h] [rbp-30h]
  _BYTE v11[25]; // [rsp+1Fh] [rbp-19h] BYREF

  v3 = *a1;
  if ( a2 == *(_QWORD *)(v3 - 16) )
  {
    result = (void *)*(unsigned int *)(v3 - 8);
    if ( (int)result <= 0 )
      return result;
    v3 = *a1;
  }
  v4 = *(_QWORD *)(v3 - 24);
  v5 = a2;
  if ( a2 < v4 )
    v5 = *(_QWORD *)(v3 - 24);
  result = sub_2215A20(v3 - 24, (__int64)v11, v5 - v4);
  v7 = *a1;
  v8 = (void *)(*a1 - 24);
  if ( v8 != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v9 = _InterlockedExchangeAdd((volatile signed __int32 *)(v7 - 8), 0xFFFFFFFF);
    }
    else
    {
      v9 = *(_DWORD *)(v7 - 8);
      *(_DWORD *)(v7 - 8) = v9 - 1;
    }
    if ( v9 <= 0 )
    {
      v10 = result;
      j_j___libc_free_0_1((unsigned __int64)v8);
      result = v10;
    }
  }
  *a1 = (__int64)result;
  return result;
}
