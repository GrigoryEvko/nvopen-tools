// Function: sub_F007D0
// Address: 0xf007d0
//
__int64 __fastcall sub_F007D0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r9
  _QWORD *v7; // rax
  __int64 v8; // r9
  _QWORD *v9; // r12
  __int64 v10; // r14
  __int64 v11; // r8
  __int64 v12; // rax
  _QWORD *v13; // rax
  unsigned __int64 v14; // rcx
  __int64 v15; // rax
  _QWORD *v16; // rax
  unsigned __int64 v17; // rcx
  __int64 v18; // rax
  _QWORD *v19; // rax
  volatile signed __int32 *v20; // r15
  unsigned int v21; // eax
  volatile signed __int32 *v22; // r13
  signed __int32 v23; // eax
  __int64 result; // rax
  signed __int32 v25; // eax
  unsigned __int64 v26; // rdx
  __int64 v27; // [rsp+10h] [rbp-40h] BYREF
  volatile signed __int32 *v28; // [rsp+18h] [rbp-38h]

  sub_EFD2C0(8u, a1 + 1576, (_DWORD *)(a1 + 1048), (__int64)"Meta", qword_497B3C8, a6);
  sub_EFCCF0(1u, a1 + 1576, a1 + 1048, (__int64)"Container info", qword_497B3A8, v6);
  v7 = (_QWORD *)sub_22077B0(544);
  v9 = v7;
  if ( v7 )
  {
    v10 = (__int64)(v7 + 2);
    v11 = 1;
    v7[1] = 0x100000001LL;
    *v7 = &unk_49D9900;
    v7[2] = v7 + 4;
    v7[3] = 0x2000000000LL;
    v12 = 0;
  }
  else
  {
    v12 = MEMORY[0x18];
    v10 = 16;
    v11 = 1;
    v26 = MEMORY[0x18] + 1LL;
    if ( MEMORY[0x1C] < v26 )
    {
      sub_C8D5F0(16, (const void *)0x20, v26, 0x10u, 1, v8);
      v12 = MEMORY[0x18];
      v11 = 1;
    }
  }
  v13 = (_QWORD *)(v9[2] + 16 * v12);
  *v13 = 1;
  v13[1] = 1;
  v14 = *((unsigned int *)v9 + 7);
  v15 = (unsigned int)(*((_DWORD *)v9 + 6) + 1);
  *((_DWORD *)v9 + 6) = v15;
  if ( v15 + 1 > v14 )
  {
    sub_C8D5F0(v10, v9 + 4, v15 + 1, 0x10u, 1, v8);
    v15 = *((unsigned int *)v9 + 6);
  }
  v16 = (_QWORD *)(v9[2] + 16 * v15);
  *v16 = 32;
  v16[1] = 2;
  v17 = *((unsigned int *)v9 + 7);
  v18 = (unsigned int)(*((_DWORD *)v9 + 6) + 1);
  *((_DWORD *)v9 + 6) = v18;
  if ( v18 + 1 > v17 )
  {
    sub_C8D5F0(v10, v9 + 4, v18 + 1, 0x10u, v11, v8);
    v18 = *((unsigned int *)v9 + 6);
  }
  v19 = (_QWORD *)(v9[2] + 16 * v18);
  v20 = (volatile signed __int32 *)(v9 + 1);
  *v19 = 2;
  v27 = v10;
  v19[1] = 2;
  v28 = (volatile signed __int32 *)v9;
  ++*((_DWORD *)v9 + 6);
  if ( &_pthread_key_create )
    _InterlockedAdd(v20, 1u);
  else
    ++*((_DWORD *)v9 + 2);
  v21 = sub_A1A630(a1 + 1576, 8, &v27);
  v22 = v28;
  *(_QWORD *)(a1 + 1736) = v21;
  if ( v22 )
  {
    if ( &_pthread_key_create )
    {
      v23 = _InterlockedExchangeAdd(v22 + 2, 0xFFFFFFFF);
    }
    else
    {
      v23 = *((_DWORD *)v22 + 2);
      *((_DWORD *)v22 + 2) = v23 - 1;
    }
    if ( v23 == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v22 + 16LL))(v22);
      if ( &_pthread_key_create )
      {
        v25 = _InterlockedExchangeAdd(v22 + 3, 0xFFFFFFFF);
      }
      else
      {
        v25 = *((_DWORD *)v22 + 3);
        *((_DWORD *)v22 + 3) = v25 - 1;
      }
      if ( v25 == 1 )
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v22 + 24LL))(v22);
    }
  }
  if ( &_pthread_key_create )
  {
    result = (unsigned int)_InterlockedExchangeAdd(v20, 0xFFFFFFFF);
    if ( (_DWORD)result != 1 )
      return result;
  }
  else
  {
    result = *((unsigned int *)v9 + 2);
    *((_DWORD *)v9 + 2) = result - 1;
    if ( (_DWORD)result != 1 )
      return result;
  }
  (*(void (__fastcall **)(_QWORD *))(*v9 + 16LL))(v9);
  if ( &_pthread_key_create )
  {
    result = (unsigned int)_InterlockedExchangeAdd((volatile signed __int32 *)v9 + 3, 0xFFFFFFFF);
  }
  else
  {
    result = *((unsigned int *)v9 + 3);
    *((_DWORD *)v9 + 3) = result - 1;
  }
  if ( (_DWORD)result == 1 )
    return (*(__int64 (__fastcall **)(_QWORD *))(*v9 + 24LL))(v9);
  return result;
}
