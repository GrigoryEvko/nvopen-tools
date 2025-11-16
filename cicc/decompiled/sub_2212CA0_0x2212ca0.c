// Function: sub_2212CA0
// Address: 0x2212ca0
//
void __fastcall sub_2212CA0(__int64 *a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rax
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r15
  unsigned __int64 v8; // r14
  __int64 v9; // r13
  __int64 v10; // rax
  __int64 v11; // r15
  unsigned __int64 v12; // r13
  __int64 v13; // rbp
  __int64 v14; // rax
  int v15; // edx
  int v16; // edx
  int v17; // edx
  _QWORD v18[8]; // [rsp+8h] [rbp-40h] BYREF

  *(_DWORD *)(a2 + 72) = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 16))(a1);
  v2 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 24))(a1);
  *(_BYTE *)(a2 + 328) = 1;
  *(_DWORD *)(a2 + 76) = v2;
  v3 = *a1;
  *(_QWORD *)(a2 + 16) = 0;
  *(_QWORD *)(a2 + 40) = 0;
  *(_QWORD *)(a2 + 56) = 0;
  (*(void (__fastcall **)(_QWORD *, __int64 *))(v3 + 32))(v18, a1);
  v4 = *(_QWORD *)(v18[0] - 24LL);
  v5 = sub_2207820(v4 + 1);
  sub_2215320(v18, v5, v4, 0);
  v6 = v18[0];
  *(_BYTE *)(v5 + v4) = 0;
  *(_QWORD *)(a2 + 16) = v5;
  *(_QWORD *)(a2 + 24) = v4;
  if ( (_UNKNOWN *)(v6 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v15 = _InterlockedExchangeAdd((volatile signed __int32 *)(v6 - 8), 0xFFFFFFFF);
    }
    else
    {
      v15 = *(_DWORD *)(v6 - 8);
      *(_DWORD *)(v6 - 8) = v15 - 1;
    }
    if ( v15 <= 0 )
      j_j___libc_free_0_1(v6 - 24);
  }
  (*(void (__fastcall **)(_QWORD *, __int64 *))(*a1 + 40))(v18, a1);
  v7 = *(_QWORD *)(v18[0] - 24LL);
  if ( (unsigned __int64)(v7 + 1) > 0x1FFFFFFFFFFFFFFELL )
    sub_426640();
  v8 = 4 * (v7 + 1);
  v9 = sub_2207820(v8);
  sub_2215FA0(v18, v9, v7, 0);
  v10 = v18[0];
  *(_QWORD *)(a2 + 40) = v9;
  *(_DWORD *)(v9 + v8 - 4) = 0;
  *(_QWORD *)(a2 + 48) = v7;
  if ( (_UNKNOWN *)(v10 - 24) != &unk_4FD67E0 )
  {
    if ( &_pthread_key_create )
    {
      v16 = _InterlockedExchangeAdd((volatile signed __int32 *)(v10 - 8), 0xFFFFFFFF);
    }
    else
    {
      v16 = *(_DWORD *)(v10 - 8);
      *(_DWORD *)(v10 - 8) = v16 - 1;
    }
    if ( v16 <= 0 )
      j_j___libc_free_0_2(v10 - 24);
  }
  (*(void (__fastcall **)(_QWORD *, __int64 *))(*a1 + 48))(v18, a1);
  v11 = *(_QWORD *)(v18[0] - 24LL);
  if ( (unsigned __int64)(v11 + 1) > 0x1FFFFFFFFFFFFFFELL )
    sub_426640();
  v12 = 4 * (v11 + 1);
  v13 = sub_2207820(v12);
  sub_2215FA0(v18, v13, v11, 0);
  v14 = v18[0];
  *(_QWORD *)(a2 + 56) = v13;
  *(_DWORD *)(v13 + v12 - 4) = 0;
  *(_QWORD *)(a2 + 64) = v11;
  if ( (_UNKNOWN *)(v14 - 24) != &unk_4FD67E0 )
  {
    if ( &_pthread_key_create )
    {
      v17 = _InterlockedExchangeAdd((volatile signed __int32 *)(v14 - 8), 0xFFFFFFFF);
    }
    else
    {
      v17 = *(_DWORD *)(v14 - 8);
      *(_DWORD *)(v14 - 8) = v17 - 1;
    }
    if ( v17 <= 0 )
      j_j___libc_free_0_2(v14 - 24);
  }
}
