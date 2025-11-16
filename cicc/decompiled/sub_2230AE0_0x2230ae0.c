// Function: sub_2230AE0
// Address: 0x2230ae0
//
void __fastcall sub_2230AE0(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // rbp
  __int64 (__fastcall *v4)(__int64); // rax
  char v5; // al
  __int64 (__fastcall *v6)(__int64); // rax
  char v7; // al
  __int64 v8; // rax
  __int64 (__fastcall *v9)(__int64); // rdx
  int v10; // edx
  void **(__fastcall *v11)(void **, __int64); // rax
  __int64 v12; // rax
  _BYTE *v13; // r12
  size_t v14; // rax
  _QWORD *v15; // rax
  unsigned __int64 v16; // rdi
  _BYTE *v17; // r15
  bool v18; // al
  void **(__fastcall *v19)(void **, __int64); // rax
  __int64 v20; // rax
  _BYTE *v21; // r14
  size_t v22; // rax
  _QWORD *v23; // rax
  unsigned __int64 v24; // rdi
  _BYTE *v25; // r14
  void **(__fastcall *v26)(void **, __int64); // rax
  __int64 v27; // rax
  _BYTE *v28; // r13
  size_t v29; // rax
  _QWORD *v30; // rax
  unsigned __int64 v31; // rdi
  _BYTE *v32; // r13
  void **(__fastcall *v33)(void **, __int64); // rax
  __int64 v34; // rax
  _BYTE *v35; // r12
  size_t v36; // rax
  _QWORD *v37; // rax
  unsigned __int64 v38; // rdi
  _BYTE *v39; // rsi
  __int64 (__fastcall *v40)(__int64); // rax
  int v41; // eax
  __int64 (__fastcall *v42)(__int64); // rax
  int v43; // eax
  _BYTE *v44; // rax
  __int64 v45; // rax
  unsigned __int64 v46; // rdi
  unsigned __int64 v47; // rdi
  unsigned __int64 v48; // rdi
  int v49; // edx
  int v50; // edx
  int v51; // edx
  int v52; // edx
  _QWORD *v54; // [rsp+40h] [rbp-58h] BYREF
  _QWORD *v55; // [rsp+48h] [rbp-50h] BYREF
  _QWORD *v56; // [rsp+50h] [rbp-48h] BYREF
  _QWORD v57[8]; // [rsp+58h] [rbp-40h] BYREF

  v3 = (_QWORD *)sub_2230A90(a2, (__int64)a2);
  v4 = *(__int64 (__fastcall **)(__int64))(*v3 + 16LL);
  if ( v4 == sub_222E840 )
    v5 = *(_BYTE *)(v3[2] + 33LL);
  else
    v5 = v4((__int64)v3);
  *(_BYTE *)(a1 + 33) = v5;
  v6 = *(__int64 (__fastcall **)(__int64))(*v3 + 24LL);
  if ( v6 == sub_222E850 )
    v7 = *(_BYTE *)(v3[2] + 34LL);
  else
    v7 = v6((__int64)v3);
  *(_BYTE *)(a1 + 34) = v7;
  v8 = *v3;
  v9 = *(__int64 (__fastcall **)(__int64))(*v3 + 64LL);
  if ( v9 == sub_222E860 )
  {
    v10 = *(_DWORD *)(v3[2] + 88LL);
  }
  else
  {
    v10 = v9((__int64)v3);
    v8 = *v3;
  }
  v11 = *(void **(__fastcall **)(void **, __int64))(v8 + 32);
  *(_DWORD *)(a1 + 88) = v10;
  if ( v11 == sub_222F170 )
  {
    v12 = v3[2];
    v13 = *(_BYTE **)(v12 + 16);
    if ( !v13 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v14 = strlen(*(const char **)(v12 + 16));
    if ( v13 == &v13[v14] )
      v15 = &unk_4FD67D8;
    else
      v15 = sub_222EC60(v13, (__int64)&v13[v14]);
    v54 = v15;
  }
  else
  {
    v11((void **)&v54, (__int64)v3);
    v15 = v54;
  }
  v16 = *(v15 - 3);
  *(_QWORD *)(a1 + 24) = v16;
  v17 = (_BYTE *)sub_2207820(v16);
  sub_2215320(&v54, v17, *(_QWORD *)(a1 + 24), 0);
  v18 = 0;
  if ( *(_QWORD *)(a1 + 24) )
    v18 = (unsigned __int8)(*v17 - 1) <= 0x7Du;
  *(_BYTE *)(a1 + 32) = v18;
  v19 = *(void **(__fastcall **)(void **, __int64))(*v3 + 40LL);
  if ( v19 == sub_222F090 )
  {
    v20 = v3[2];
    v21 = *(_BYTE **)(v20 + 40);
    if ( !v21 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v22 = strlen(*(const char **)(v20 + 40));
    if ( v21 == &v21[v22] )
      v23 = &unk_4FD67D8;
    else
      v23 = sub_222EC60(v21, (__int64)&v21[v22]);
    v55 = v23;
  }
  else
  {
    v19((void **)&v55, (__int64)v3);
    v23 = v55;
  }
  v24 = *(v23 - 3);
  *(_QWORD *)(a1 + 48) = v24;
  v25 = (_BYTE *)sub_2207820(v24);
  sub_2215320(&v55, v25, *(_QWORD *)(a1 + 48), 0);
  v26 = *(void **(__fastcall **)(void **, __int64))(*v3 + 48LL);
  if ( v26 == sub_222F100 )
  {
    v27 = v3[2];
    v28 = *(_BYTE **)(v27 + 56);
    if ( !v28 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v29 = strlen(*(const char **)(v27 + 56));
    if ( v28 == &v28[v29] )
      v30 = &unk_4FD67D8;
    else
      v30 = sub_222EC60(v28, (__int64)&v28[v29]);
    v56 = v30;
  }
  else
  {
    v26((void **)&v56, (__int64)v3);
    v30 = v56;
  }
  v31 = *(v30 - 3);
  *(_QWORD *)(a1 + 64) = v31;
  v32 = (_BYTE *)sub_2207820(v31);
  sub_2215320(&v56, v32, *(_QWORD *)(a1 + 64), 0);
  v33 = *(void **(__fastcall **)(void **, __int64))(*v3 + 56LL);
  if ( v33 == sub_222F020 )
  {
    v34 = v3[2];
    v35 = *(_BYTE **)(v34 + 72);
    if ( !v35 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v36 = strlen(*(const char **)(v34 + 72));
    if ( v35 == &v35[v36] )
      v37 = &unk_4FD67D8;
    else
      v37 = sub_222EC60(v35, (__int64)&v35[v36]);
    v57[0] = v37;
  }
  else
  {
    v33((void **)v57, (__int64)v3);
    v37 = (_QWORD *)v57[0];
  }
  v38 = *(v37 - 3);
  *(_QWORD *)(a1 + 80) = v38;
  v39 = (_BYTE *)sub_2207820(v38);
  sub_2215320(v57, v39, *(_QWORD *)(a1 + 80), 0);
  v40 = *(__int64 (__fastcall **)(__int64))(*v3 + 72LL);
  if ( v40 == sub_222E870 )
    v41 = *(_DWORD *)(v3[2] + 92LL);
  else
    v41 = v40((__int64)v3);
  *(_DWORD *)(a1 + 92) = v41;
  v42 = *(__int64 (__fastcall **)(__int64))(*v3 + 80LL);
  if ( v42 == sub_222E880 )
    v43 = *(_DWORD *)(v3[2] + 96LL);
  else
    v43 = v42((__int64)v3);
  *(_DWORD *)(a1 + 96) = v43;
  v44 = (_BYTE *)sub_222F790(a2, (__int64)v39);
  sub_222F5F0(v44, off_4CDFAD0, off_4CDFAD0 + 11, (void *)(a1 + 100));
  v45 = v57[0];
  *(_QWORD *)(a1 + 16) = v17;
  *(_QWORD *)(a1 + 40) = v25;
  *(_QWORD *)(a1 + 56) = v32;
  *(_QWORD *)(a1 + 72) = v39;
  *(_BYTE *)(a1 + 111) = 1;
  if ( (_UNKNOWN *)(v45 - 24) != &unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v49 = _InterlockedExchangeAdd((volatile signed __int32 *)(v45 - 8), 0xFFFFFFFF);
    }
    else
    {
      v49 = *(_DWORD *)(v45 - 8);
      *(_DWORD *)(v45 - 8) = v49 - 1;
    }
    if ( v49 <= 0 )
      j_j___libc_free_0_1(v45 - 24);
  }
  v46 = (unsigned __int64)(v56 - 3);
  if ( v56 - 3 != (_QWORD *)&unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v50 = _InterlockedExchangeAdd((volatile signed __int32 *)v56 - 2, 0xFFFFFFFF);
    }
    else
    {
      v50 = *((_DWORD *)v56 - 2);
      *((_DWORD *)v56 - 2) = v50 - 1;
    }
    if ( v50 <= 0 )
      j_j___libc_free_0_1(v46);
  }
  v47 = (unsigned __int64)(v55 - 3);
  if ( v55 - 3 != (_QWORD *)&unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v51 = _InterlockedExchangeAdd((volatile signed __int32 *)v55 - 2, 0xFFFFFFFF);
    }
    else
    {
      v51 = *((_DWORD *)v55 - 2);
      *((_DWORD *)v55 - 2) = v51 - 1;
    }
    if ( v51 <= 0 )
      j_j___libc_free_0_1(v47);
  }
  v48 = (unsigned __int64)(v54 - 3);
  if ( v54 - 3 != (_QWORD *)&unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v52 = _InterlockedExchangeAdd((volatile signed __int32 *)v54 - 2, 0xFFFFFFFF);
    }
    else
    {
      v52 = *((_DWORD *)v54 - 2);
      *((_DWORD *)v54 - 2) = v52 - 1;
    }
    if ( v52 <= 0 )
      j_j___libc_free_0_1(v48);
  }
}
