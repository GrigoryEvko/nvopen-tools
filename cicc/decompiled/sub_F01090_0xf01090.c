// Function: sub_F01090
// Address: 0xf01090
//
__int64 __fastcall sub_F01090(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  _QWORD *v6; // rax
  __int64 v7; // r9
  _QWORD *v8; // r12
  __int64 v9; // r14
  __int64 v10; // rax
  _QWORD *v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rax
  _QWORD *v14; // rax
  volatile signed __int32 *v15; // r15
  unsigned int v16; // eax
  volatile signed __int32 *v17; // r13
  signed __int32 v18; // eax
  __int64 result; // rax
  signed __int32 v20; // eax
  unsigned __int64 v21; // rdx
  __int64 v22; // [rsp+10h] [rbp-40h] BYREF
  volatile signed __int32 *v23; // [rsp+18h] [rbp-38h]

  sub_EFCCF0(4u, a1 + 1576, a1 + 1048, (__int64)"External File", qword_497B378, a6);
  v6 = (_QWORD *)sub_22077B0(544);
  v8 = v6;
  if ( v6 )
  {
    v9 = (__int64)(v6 + 2);
    v6[1] = 0x100000001LL;
    *v6 = &unk_49D9900;
    v6[2] = v6 + 4;
    v6[3] = 0x2000000000LL;
    v10 = 0;
  }
  else
  {
    v10 = MEMORY[0x18];
    v9 = 16;
    v21 = MEMORY[0x18] + 1LL;
    if ( v21 > MEMORY[0x1C] )
    {
      sub_C8D5F0(16, (const void *)0x20, v21, 0x10u, 4, v7);
      v10 = MEMORY[0x18];
    }
  }
  v11 = (_QWORD *)(v8[2] + 16 * v10);
  *v11 = 4;
  v11[1] = 1;
  v12 = *((unsigned int *)v8 + 7);
  v13 = (unsigned int)(*((_DWORD *)v8 + 6) + 1);
  *((_DWORD *)v8 + 6) = v13;
  if ( v13 + 1 > v12 )
  {
    sub_C8D5F0(v9, v8 + 4, v13 + 1, 0x10u, 4, v7);
    v13 = *((unsigned int *)v8 + 6);
  }
  v14 = (_QWORD *)(v8[2] + 16 * v13);
  v15 = (volatile signed __int32 *)(v8 + 1);
  *v14 = 0;
  v22 = v9;
  v14[1] = 10;
  v23 = (volatile signed __int32 *)v8;
  ++*((_DWORD *)v8 + 6);
  if ( &_pthread_key_create )
    _InterlockedAdd(v15, 1u);
  else
    ++*((_DWORD *)v8 + 2);
  v16 = sub_A1A630(a1 + 1576, 8, &v22);
  v17 = v23;
  *(_QWORD *)(a1 + 1760) = v16;
  if ( v17 )
  {
    if ( &_pthread_key_create )
    {
      v18 = _InterlockedExchangeAdd(v17 + 2, 0xFFFFFFFF);
    }
    else
    {
      v18 = *((_DWORD *)v17 + 2);
      *((_DWORD *)v17 + 2) = v18 - 1;
    }
    if ( v18 == 1 )
    {
      (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v17 + 16LL))(v17);
      if ( &_pthread_key_create )
      {
        v20 = _InterlockedExchangeAdd(v17 + 3, 0xFFFFFFFF);
      }
      else
      {
        v20 = *((_DWORD *)v17 + 3);
        *((_DWORD *)v17 + 3) = v20 - 1;
      }
      if ( v20 == 1 )
        (*(void (__fastcall **)(volatile signed __int32 *))(*(_QWORD *)v17 + 24LL))(v17);
    }
  }
  if ( &_pthread_key_create )
  {
    result = (unsigned int)_InterlockedExchangeAdd(v15, 0xFFFFFFFF);
    if ( (_DWORD)result != 1 )
      return result;
  }
  else
  {
    result = *((unsigned int *)v8 + 2);
    *((_DWORD *)v8 + 2) = result - 1;
    if ( (_DWORD)result != 1 )
      return result;
  }
  (*(void (__fastcall **)(_QWORD *))(*v8 + 16LL))(v8);
  if ( &_pthread_key_create )
  {
    result = (unsigned int)_InterlockedExchangeAdd((volatile signed __int32 *)v8 + 3, 0xFFFFFFFF);
  }
  else
  {
    result = *((unsigned int *)v8 + 3);
    *((_DWORD *)v8 + 3) = result - 1;
  }
  if ( (_DWORD)result == 1 )
    return (*(__int64 (__fastcall **)(_QWORD *))(*v8 + 24LL))(v8);
  return result;
}
