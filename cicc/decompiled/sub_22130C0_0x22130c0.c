// Function: sub_22130C0
// Address: 0x22130c0
//
__int64 __fastcall sub_22130C0(__int64 *a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // r15
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // r15
  __int64 v14; // r13
  __int64 v15; // rax
  __int64 result; // rax
  int v17; // edx
  int v18; // edx
  int v19; // edx
  int v20; // edx
  _QWORD v21[8]; // [rsp+8h] [rbp-40h] BYREF

  *(_BYTE *)(a2 + 33) = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 16))(a1);
  *(_BYTE *)(a2 + 34) = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 24))(a1);
  v2 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 64))(a1);
  *(_BYTE *)(a2 + 111) = 1;
  *(_DWORD *)(a2 + 88) = v2;
  v3 = *a1;
  *(_QWORD *)(a2 + 16) = 0;
  *(_QWORD *)(a2 + 40) = 0;
  *(_QWORD *)(a2 + 56) = 0;
  *(_QWORD *)(a2 + 72) = 0;
  (*(void (__fastcall **)(_QWORD *, __int64 *))(v3 + 32))(v21, a1);
  v4 = *(_QWORD *)(v21[0] - 24LL);
  v5 = sub_2207820(v4 + 1);
  sub_2215320(v21, v5, v4, 0);
  v6 = v21[0];
  *(_BYTE *)(v5 + v4) = 0;
  *(_QWORD *)(a2 + 24) = v4;
  *(_QWORD *)(a2 + 16) = v5;
  if ( (_UNKNOWN *)(v6 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v17 = _InterlockedExchangeAdd((volatile signed __int32 *)(v6 - 8), 0xFFFFFFFF);
    }
    else
    {
      v17 = *(_DWORD *)(v6 - 8);
      *(_DWORD *)(v6 - 8) = v17 - 1;
    }
    if ( v17 <= 0 )
      j_j___libc_free_0_1(v6 - 24);
  }
  (*(void (__fastcall **)(_QWORD *, __int64 *))(*a1 + 40))(v21, a1);
  v7 = *(_QWORD *)(v21[0] - 24LL);
  v8 = sub_2207820(v7 + 1);
  sub_2215320(v21, v8, v7, 0);
  v9 = v21[0];
  *(_BYTE *)(v8 + v7) = 0;
  *(_QWORD *)(a2 + 40) = v8;
  *(_QWORD *)(a2 + 48) = v7;
  if ( (_UNKNOWN *)(v9 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v18 = _InterlockedExchangeAdd((volatile signed __int32 *)(v9 - 8), 0xFFFFFFFF);
    }
    else
    {
      v18 = *(_DWORD *)(v9 - 8);
      *(_DWORD *)(v9 - 8) = v18 - 1;
    }
    if ( v18 <= 0 )
      j_j___libc_free_0_1(v9 - 24);
  }
  (*(void (__fastcall **)(_QWORD *, __int64 *))(*a1 + 48))(v21, a1);
  v10 = *(_QWORD *)(v21[0] - 24LL);
  v11 = sub_2207820(v10 + 1);
  sub_2215320(v21, v11, v10, 0);
  v12 = v21[0];
  *(_BYTE *)(v11 + v10) = 0;
  *(_QWORD *)(a2 + 56) = v11;
  *(_QWORD *)(a2 + 64) = v10;
  if ( (_UNKNOWN *)(v12 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v19 = _InterlockedExchangeAdd((volatile signed __int32 *)(v12 - 8), 0xFFFFFFFF);
    }
    else
    {
      v19 = *(_DWORD *)(v12 - 8);
      *(_DWORD *)(v12 - 8) = v19 - 1;
    }
    if ( v19 <= 0 )
      j_j___libc_free_0_1(v12 - 24);
  }
  (*(void (__fastcall **)(_QWORD *, __int64 *))(*a1 + 56))(v21, a1);
  v13 = *(_QWORD *)(v21[0] - 24LL);
  v14 = sub_2207820(v13 + 1);
  sub_2215320(v21, v14, v13, 0);
  v15 = v21[0];
  *(_BYTE *)(v14 + v13) = 0;
  *(_QWORD *)(a2 + 72) = v14;
  *(_QWORD *)(a2 + 80) = v13;
  if ( (_UNKNOWN *)(v15 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v20 = _InterlockedExchangeAdd((volatile signed __int32 *)(v15 - 8), 0xFFFFFFFF);
    }
    else
    {
      v20 = *(_DWORD *)(v15 - 8);
      *(_DWORD *)(v15 - 8) = v20 - 1;
    }
    if ( v20 <= 0 )
      j_j___libc_free_0_1(v15 - 24);
  }
  *(_DWORD *)(a2 + 92) = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 72))(a1);
  result = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 80))(a1);
  *(_DWORD *)(a2 + 96) = result;
  return result;
}
