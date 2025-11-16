// Function: sub_22127C0
// Address: 0x22127c0
//
__int64 __fastcall sub_22127C0(
        __int64 a1,
        int a2,
        int a3,
        int a4,
        int a5,
        unsigned __int8 a6,
        __int64 a7,
        _DWORD *a8,
        __int64 a9)
{
  __int64 v9; // rdi
  __int64 v10; // r12
  unsigned __int64 v12; // rdi
  int v13; // edx
  char v14; // [rsp+13h] [rbp-6Dh] BYREF
  int v15; // [rsp+14h] [rbp-6Ch] BYREF
  __int64 v16; // [rsp+18h] [rbp-68h] BYREF
  _QWORD v17[4]; // [rsp+20h] [rbp-60h] BYREF
  void (__fastcall *v18)(_QWORD *); // [rsp+40h] [rbp-40h]

  v9 = *(_QWORD *)(a1 + 16);
  v18 = 0;
  v15 = 0;
  v10 = sub_2222740(v9, a2, a3, a4, a5, a6, a7, (__int64)&v15, 0, (__int64)v17);
  if ( v15 )
  {
    *a8 = v15;
  }
  else
  {
    if ( !v18 )
      sub_426248((__int64)"uninitialized __any_string");
    sub_2216C30(&v16, v17[0], v17[1], &v14);
    sub_2216010(a9, &v16);
    v12 = v16 - 24;
    if ( (_UNKNOWN *)(v16 - 24) != &unk_4FD67E0 )
    {
      if ( &_pthread_key_create )
      {
        v13 = _InterlockedExchangeAdd((volatile signed __int32 *)(v16 - 8), 0xFFFFFFFF);
      }
      else
      {
        v13 = *(_DWORD *)(v16 - 8);
        *(_DWORD *)(v16 - 8) = v13 - 1;
      }
      if ( v13 <= 0 )
        j_j___libc_free_0_2(v12);
    }
  }
  if ( v18 )
    v18(v17);
  return v10;
}
