// Function: sub_2213A40
// Address: 0x2213a40
//
__int64 __fastcall sub_2213A40(__int64 *a1, __int64 a2)
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
  unsigned __int64 v12; // r14
  __int64 v13; // r13
  __int64 v14; // rax
  __int64 v15; // r15
  unsigned __int64 v16; // r14
  __int64 v17; // r13
  __int64 v18; // rax
  __int64 result; // rax
  int v20; // edx
  int v21; // edx
  int v22; // edx
  int v23; // edx
  _QWORD v24[8]; // [rsp+8h] [rbp-40h] BYREF

  *(_DWORD *)(a2 + 36) = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 16))(a1);
  *(_DWORD *)(a2 + 40) = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 24))(a1);
  v2 = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 64))(a1);
  *(_QWORD *)(a2 + 16) = 0;
  *(_DWORD *)(a2 + 96) = v2;
  v3 = *a1;
  *(_QWORD *)(a2 + 48) = 0;
  *(_QWORD *)(a2 + 64) = 0;
  *(_QWORD *)(a2 + 80) = 0;
  *(_BYTE *)(a2 + 152) = 1;
  (*(void (__fastcall **)(_QWORD *, __int64 *))(v3 + 32))(v24, a1);
  v4 = *(_QWORD *)(v24[0] - 24LL);
  v5 = sub_2207820(v4 + 1);
  sub_2215320(v24, v5, v4, 0);
  v6 = v24[0];
  *(_BYTE *)(v5 + v4) = 0;
  *(_QWORD *)(a2 + 16) = v5;
  *(_QWORD *)(a2 + 24) = v4;
  if ( (_UNKNOWN *)(v6 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v20 = _InterlockedExchangeAdd((volatile signed __int32 *)(v6 - 8), 0xFFFFFFFF);
    }
    else
    {
      v20 = *(_DWORD *)(v6 - 8);
      *(_DWORD *)(v6 - 8) = v20 - 1;
    }
    if ( v20 <= 0 )
      j_j___libc_free_0_1(v6 - 24);
  }
  (*(void (__fastcall **)(_QWORD *, __int64 *))(*a1 + 40))(v24, a1);
  v7 = *(_QWORD *)(v24[0] - 24LL);
  if ( (unsigned __int64)(v7 + 1) > 0x1FFFFFFFFFFFFFFELL )
    sub_426640();
  v8 = 4 * (v7 + 1);
  v9 = sub_2207820(v8);
  sub_2215FA0(v24, v9, v7, 0);
  v10 = v24[0];
  *(_QWORD *)(a2 + 48) = v9;
  *(_DWORD *)(v9 + v8 - 4) = 0;
  *(_QWORD *)(a2 + 56) = v7;
  if ( (_UNKNOWN *)(v10 - 24) != &unk_4FD67E0 )
  {
    if ( &_pthread_key_create )
    {
      v21 = _InterlockedExchangeAdd((volatile signed __int32 *)(v10 - 8), 0xFFFFFFFF);
    }
    else
    {
      v21 = *(_DWORD *)(v10 - 8);
      *(_DWORD *)(v10 - 8) = v21 - 1;
    }
    if ( v21 <= 0 )
      j_j___libc_free_0_2(v10 - 24);
  }
  (*(void (__fastcall **)(_QWORD *, __int64 *))(*a1 + 48))(v24, a1);
  v11 = *(_QWORD *)(v24[0] - 24LL);
  if ( (unsigned __int64)(v11 + 1) > 0x1FFFFFFFFFFFFFFELL )
    sub_426640();
  v12 = 4 * (v11 + 1);
  v13 = sub_2207820(v12);
  sub_2215FA0(v24, v13, v11, 0);
  v14 = v24[0];
  *(_QWORD *)(a2 + 64) = v13;
  *(_DWORD *)(v13 + v12 - 4) = 0;
  *(_QWORD *)(a2 + 72) = v11;
  if ( (_UNKNOWN *)(v14 - 24) != &unk_4FD67E0 )
  {
    if ( &_pthread_key_create )
    {
      v22 = _InterlockedExchangeAdd((volatile signed __int32 *)(v14 - 8), 0xFFFFFFFF);
    }
    else
    {
      v22 = *(_DWORD *)(v14 - 8);
      *(_DWORD *)(v14 - 8) = v22 - 1;
    }
    if ( v22 <= 0 )
      j_j___libc_free_0_2(v14 - 24);
  }
  (*(void (__fastcall **)(_QWORD *, __int64 *))(*a1 + 56))(v24, a1);
  v15 = *(_QWORD *)(v24[0] - 24LL);
  if ( (unsigned __int64)(v15 + 1) > 0x1FFFFFFFFFFFFFFELL )
    sub_426640();
  v16 = 4 * (v15 + 1);
  v17 = sub_2207820(v16);
  sub_2215FA0(v24, v17, v15, 0);
  v18 = v24[0];
  *(_QWORD *)(a2 + 80) = v17;
  *(_DWORD *)(v17 + v16 - 4) = 0;
  *(_QWORD *)(a2 + 88) = v15;
  if ( (_UNKNOWN *)(v18 - 24) != &unk_4FD67E0 )
  {
    if ( &_pthread_key_create )
    {
      v23 = _InterlockedExchangeAdd((volatile signed __int32 *)(v18 - 8), 0xFFFFFFFF);
    }
    else
    {
      v23 = *(_DWORD *)(v18 - 8);
      *(_DWORD *)(v18 - 8) = v23 - 1;
    }
    if ( v23 <= 0 )
      j_j___libc_free_0_2(v18 - 24);
  }
  *(_DWORD *)(a2 + 100) = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 72))(a1);
  result = (*(__int64 (__fastcall **)(__int64 *))(*a1 + 80))(a1);
  *(_DWORD *)(a2 + 104) = result;
  return result;
}
