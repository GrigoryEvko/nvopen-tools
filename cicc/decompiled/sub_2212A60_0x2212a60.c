// Function: sub_2212A60
// Address: 0x2212a60
//
void __fastcall sub_2212A60(__int64 *a1, __int64 a2)
{
  char v2; // al
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r15
  __int64 v8; // r14
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rbp
  __int64 v12; // rax
  int v13; // edx
  int v14; // edx
  int v15; // edx
  _QWORD v16[8]; // [rsp+8h] [rbp-40h] BYREF

  *(_BYTE *)(a2 + 72) = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 16))(a1);
  v2 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 24))(a1);
  *(_BYTE *)(a2 + 136) = 1;
  *(_BYTE *)(a2 + 73) = v2;
  v3 = *a1;
  *(_QWORD *)(a2 + 16) = 0;
  *(_QWORD *)(a2 + 40) = 0;
  *(_QWORD *)(a2 + 56) = 0;
  (*(void (__fastcall **)(_QWORD *, __int64 *))(v3 + 32))(v16, a1);
  v4 = *(_QWORD *)(v16[0] - 24LL);
  v5 = sub_2207820(v4 + 1);
  sub_2215320(v16, v5, v4, 0);
  v6 = v16[0];
  *(_BYTE *)(v5 + v4) = 0;
  *(_QWORD *)(a2 + 16) = v5;
  *(_QWORD *)(a2 + 24) = v4;
  if ( (_UNKNOWN *)(v6 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v13 = _InterlockedExchangeAdd((volatile signed __int32 *)(v6 - 8), 0xFFFFFFFF);
    }
    else
    {
      v13 = *(_DWORD *)(v6 - 8);
      *(_DWORD *)(v6 - 8) = v13 - 1;
    }
    if ( v13 <= 0 )
      j_j___libc_free_0_1(v6 - 24);
  }
  (*(void (__fastcall **)(_QWORD *, __int64 *))(*a1 + 40))(v16, a1);
  v7 = *(_QWORD *)(v16[0] - 24LL);
  v8 = sub_2207820(v7 + 1);
  sub_2215320(v16, v8, v7, 0);
  v9 = v16[0];
  *(_BYTE *)(v8 + v7) = 0;
  *(_QWORD *)(a2 + 40) = v8;
  *(_QWORD *)(a2 + 48) = v7;
  if ( (_UNKNOWN *)(v9 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v14 = _InterlockedExchangeAdd((volatile signed __int32 *)(v9 - 8), 0xFFFFFFFF);
    }
    else
    {
      v14 = *(_DWORD *)(v9 - 8);
      *(_DWORD *)(v9 - 8) = v14 - 1;
    }
    if ( v14 <= 0 )
      j_j___libc_free_0_1(v9 - 24);
  }
  (*(void (__fastcall **)(_QWORD *, __int64 *))(*a1 + 48))(v16, a1);
  v10 = *(_QWORD *)(v16[0] - 24LL);
  v11 = sub_2207820(v10 + 1);
  sub_2215320(v16, v11, v10, 0);
  v12 = v16[0];
  *(_BYTE *)(v11 + v10) = 0;
  *(_QWORD *)(a2 + 56) = v11;
  *(_QWORD *)(a2 + 64) = v10;
  if ( (_UNKNOWN *)(v12 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v15 = _InterlockedExchangeAdd((volatile signed __int32 *)(v12 - 8), 0xFFFFFFFF);
    }
    else
    {
      v15 = *(_DWORD *)(v12 - 8);
      *(_DWORD *)(v12 - 8) = v15 - 1;
    }
    if ( v15 <= 0 )
      j_j___libc_free_0_1(v12 - 24);
  }
}
