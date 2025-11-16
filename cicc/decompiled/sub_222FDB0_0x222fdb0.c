// Function: sub_222FDB0
// Address: 0x222fdb0
//
void __fastcall sub_222FDB0(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // rbp
  void **(__fastcall *v4)(void **, __int64); // rax
  __int64 v5; // rax
  _BYTE *v6; // r12
  size_t v7; // rax
  _QWORD *v8; // rax
  unsigned __int64 v9; // rdi
  _BYTE *v10; // r14
  bool v11; // al
  void **(__fastcall *v12)(void **, __int64); // rax
  __int64 v13; // rax
  _BYTE *v14; // r13
  size_t v15; // rax
  _QWORD *v16; // rax
  unsigned __int64 v17; // rdi
  _BYTE *v18; // r13
  void **(__fastcall *v19)(void **, __int64); // rax
  __int64 v20; // rax
  _BYTE *v21; // r12
  size_t v22; // rax
  _QWORD *v23; // rax
  unsigned __int64 v24; // rdi
  _BYTE *v25; // rsi
  __int64 (__fastcall *v26)(__int64); // rax
  char v27; // al
  __int64 (__fastcall *v28)(__int64); // rax
  char v29; // al
  _BYTE *v30; // rbp
  __int64 v31; // rax
  unsigned __int64 v32; // rdi
  unsigned __int64 v33; // rdi
  int v34; // edx
  int v35; // edx
  int v36; // edx
  _QWORD *v38; // [rsp+28h] [rbp-50h] BYREF
  _QWORD *v39; // [rsp+30h] [rbp-48h] BYREF
  _QWORD v40[8]; // [rsp+38h] [rbp-40h] BYREF

  v3 = (_QWORD *)sub_222FD60(a2, (__int64)a2);
  v4 = *(void **(__fastcall **)(void **, __int64))(*v3 + 32LL);
  if ( v4 == sub_222ED80 )
  {
    v5 = v3[2];
    v6 = *(_BYTE **)(v5 + 16);
    if ( !v6 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v7 = strlen(*(const char **)(v5 + 16));
    if ( v6 == &v6[v7] )
      v8 = &unk_4FD67D8;
    else
      v8 = sub_222EC60(v6, (__int64)&v6[v7]);
    v38 = v8;
  }
  else
  {
    v4((void **)&v38, (__int64)v3);
    v8 = v38;
  }
  v9 = *(v8 - 3);
  *(_QWORD *)(a1 + 24) = v9;
  v10 = (_BYTE *)sub_2207820(v9);
  sub_2215320(&v38, v10, *(_QWORD *)(a1 + 24), 0);
  v11 = 0;
  if ( *(_QWORD *)(a1 + 24) )
    v11 = (unsigned __int8)(*v10 - 1) <= 0x7Du;
  *(_BYTE *)(a1 + 32) = v11;
  v12 = *(void **(__fastcall **)(void **, __int64))(*v3 + 40LL);
  if ( v12 == sub_222EDF0 )
  {
    v13 = v3[2];
    v14 = *(_BYTE **)(v13 + 40);
    if ( !v14 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v15 = strlen(*(const char **)(v13 + 40));
    if ( v14 == &v14[v15] )
      v16 = &unk_4FD67D8;
    else
      v16 = sub_222EC60(v14, (__int64)&v14[v15]);
    v39 = v16;
  }
  else
  {
    v12((void **)&v39, (__int64)v3);
    v16 = v39;
  }
  v17 = *(v16 - 3);
  *(_QWORD *)(a1 + 48) = v17;
  v18 = (_BYTE *)sub_2207820(v17);
  sub_2215320(&v39, v18, *(_QWORD *)(a1 + 48), 0);
  v19 = *(void **(__fastcall **)(void **, __int64))(*v3 + 48LL);
  if ( v19 == sub_222ED10 )
  {
    v20 = v3[2];
    v21 = *(_BYTE **)(v20 + 56);
    if ( !v21 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v22 = strlen(*(const char **)(v20 + 56));
    if ( v21 == &v21[v22] )
      v23 = &unk_4FD67D8;
    else
      v23 = sub_222EC60(v21, (__int64)&v21[v22]);
    v40[0] = v23;
  }
  else
  {
    v19((void **)v40, (__int64)v3);
    v23 = (_QWORD *)v40[0];
  }
  v24 = *(v23 - 3);
  *(_QWORD *)(a1 + 64) = v24;
  v25 = (_BYTE *)sub_2207820(v24);
  sub_2215320(v40, v25, *(_QWORD *)(a1 + 64), 0);
  v26 = *(__int64 (__fastcall **)(__int64))(*v3 + 16LL);
  if ( v26 == sub_222E8E0 )
    v27 = *(_BYTE *)(v3[2] + 72LL);
  else
    v27 = v26((__int64)v3);
  *(_BYTE *)(a1 + 72) = v27;
  v28 = *(__int64 (__fastcall **)(__int64))(*v3 + 24LL);
  if ( v28 == sub_222E8F0 )
    v29 = *(_BYTE *)(v3[2] + 73LL);
  else
    v29 = v28((__int64)v3);
  *(_BYTE *)(a1 + 73) = v29;
  v30 = (_BYTE *)sub_222F790(a2, (__int64)v25);
  sub_222F5F0(v30, off_4CDFAC0[0], off_4CDFAC0[0] + 36, (void *)(a1 + 74));
  sub_222F5F0(v30, off_4CDFAC8[0], off_4CDFAC8[0] + 26, (void *)(a1 + 110));
  v31 = v40[0];
  *(_QWORD *)(a1 + 16) = v10;
  *(_QWORD *)(a1 + 40) = v18;
  *(_QWORD *)(a1 + 56) = v25;
  *(_BYTE *)(a1 + 136) = 1;
  if ( (_UNKNOWN *)(v31 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v36 = _InterlockedExchangeAdd((volatile signed __int32 *)(v31 - 8), 0xFFFFFFFF);
    }
    else
    {
      v36 = *(_DWORD *)(v31 - 8);
      *(_DWORD *)(v31 - 8) = v36 - 1;
    }
    if ( v36 <= 0 )
      j_j___libc_free_0_1(v31 - 24);
  }
  v32 = (unsigned __int64)(v39 - 3);
  if ( v39 - 3 != (_QWORD *)&unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v34 = _InterlockedExchangeAdd((volatile signed __int32 *)v39 - 2, 0xFFFFFFFF);
    }
    else
    {
      v34 = *((_DWORD *)v39 - 2);
      *((_DWORD *)v39 - 2) = v34 - 1;
    }
    if ( v34 <= 0 )
      j_j___libc_free_0_1(v32);
  }
  v33 = (unsigned __int64)(v38 - 3);
  if ( v38 - 3 != (_QWORD *)&unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v35 = _InterlockedExchangeAdd((volatile signed __int32 *)v38 - 2, 0xFFFFFFFF);
    }
    else
    {
      v35 = *((_DWORD *)v38 - 2);
      *((_DWORD *)v38 - 2) = v35 - 1;
    }
    if ( v35 <= 0 )
      j_j___libc_free_0_1(v33);
  }
}
