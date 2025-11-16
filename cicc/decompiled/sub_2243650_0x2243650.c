// Function: sub_2243650
// Address: 0x2243650
//
void __fastcall sub_2243650(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // rbp
  _QWORD *(__fastcall *v4)(_QWORD *, __int64); // rax
  __int64 v5; // rax
  __int64 v6; // rsi
  _BYTE *v7; // r12
  _QWORD *v8; // rax
  unsigned __int64 v9; // rdi
  _BYTE *v10; // r14
  bool v11; // al
  __int64 *(__fastcall *v12)(__int64 *, __int64); // rax
  __int64 v13; // rax
  wchar_t *v14; // r13
  __int64 v15; // rsi
  _QWORD *v16; // rax
  unsigned __int64 v17; // rdi
  wchar_t *v18; // r13
  __int64 *(__fastcall *v19)(__int64 *, __int64); // rax
  __int64 v20; // rax
  wchar_t *v21; // r12
  __int64 v22; // rsi
  _QWORD *v23; // rax
  unsigned __int64 v24; // rdi
  wchar_t *v25; // rsi
  __int64 (__fastcall *v26)(__int64); // rdx
  __int64 (__fastcall *v27)(__int64); // rax
  int v28; // eax
  __int64 v29; // rbp
  __int64 v30; // rax
  unsigned __int64 v31; // rdi
  unsigned __int64 v32; // rdi
  int v33; // edx
  __int64 v34; // rax
  int v35; // edx
  int v36; // edx
  int v37; // edx
  _QWORD *v39; // [rsp+28h] [rbp-50h] BYREF
  _QWORD *v40; // [rsp+30h] [rbp-48h] BYREF
  __int64 v41[8]; // [rsp+38h] [rbp-40h] BYREF

  v3 = (_QWORD *)sub_2243600(a2, (__int64)a2);
  v4 = *(_QWORD *(__fastcall **)(_QWORD *, __int64))(*v3 + 32LL);
  if ( v4 == sub_2242660 )
  {
    v5 = v3[2];
    v6 = -1;
    v7 = *(_BYTE **)(v5 + 16);
    if ( v7 )
      v6 = (__int64)&v7[strlen(*(const char **)(v5 + 16))];
    v8 = sub_22424F0(v7, (_BYTE *)v6);
    v39 = v8;
  }
  else
  {
    v4(&v39, (__int64)v3);
    v8 = v39;
  }
  v9 = *(v8 - 3);
  *(_QWORD *)(a1 + 24) = v9;
  v10 = (_BYTE *)sub_2207820(v9);
  sub_2215320(&v39, v10, *(_QWORD *)(a1 + 24), 0);
  v11 = 0;
  if ( *(_QWORD *)(a1 + 24) )
    v11 = (unsigned __int8)(*v10 - 1) <= 0x7Du;
  *(_BYTE *)(a1 + 32) = v11;
  v12 = *(__int64 *(__fastcall **)(__int64 *, __int64))(*v3 + 40LL);
  if ( v12 == sub_2242AD0 )
  {
    v13 = v3[2];
    v14 = *(wchar_t **)(v13 + 40);
    if ( !v14 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v15 = (__int64)&v14[wcslen(*(const wchar_t **)(v13 + 40))];
    if ( v14 == (wchar_t *)v15 )
      v16 = &unk_4FD67F8;
    else
      v16 = (_QWORD *)sub_2242590(v14, v15);
    v40 = v16;
  }
  else
  {
    v15 = (__int64)v3;
    v12((__int64 *)&v40, (__int64)v3);
    v16 = v40;
  }
  v17 = *(v16 - 3);
  *(_QWORD *)(a1 + 48) = v17;
  if ( v17 > 0x1FFFFFFFFFFFFFFELL )
    sub_426640(v17, v15);
  v18 = (wchar_t *)sub_2207820(4 * v17);
  sub_2215FA0((__int64 *)&v40, v18, *(_QWORD *)(a1 + 48), 0);
  v19 = *(__int64 *(__fastcall **)(__int64 *, __int64))(*v3 + 48LL);
  if ( v19 == sub_2242B40 )
  {
    v20 = v3[2];
    v21 = *(wchar_t **)(v20 + 56);
    if ( !v21 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v22 = (__int64)&v21[wcslen(*(const wchar_t **)(v20 + 56))];
    if ( v21 == (wchar_t *)v22 )
      v23 = &unk_4FD67F8;
    else
      v23 = (_QWORD *)sub_2242590(v21, v22);
    v41[0] = (__int64)v23;
  }
  else
  {
    v22 = (__int64)v3;
    v19(v41, (__int64)v3);
    v23 = (_QWORD *)v41[0];
  }
  v24 = *(v23 - 3);
  *(_QWORD *)(a1 + 64) = v24;
  if ( v24 > 0x1FFFFFFFFFFFFFFELL )
    sub_426640(v24, v22);
  v25 = (wchar_t *)sub_2207820(4 * v24);
  sub_2215FA0(v41, v25, *(_QWORD *)(a1 + 64), 0);
  v26 = *(__int64 (__fastcall **)(__int64))(*v3 + 16LL);
  if ( v26 != sub_22421B0 )
  {
    v33 = v26((__int64)v3);
    v34 = *v3;
    *(_DWORD *)(a1 + 72) = v33;
    v27 = *(__int64 (__fastcall **)(__int64))(v34 + 24);
    if ( v27 == sub_22421C0 )
      goto LABEL_21;
LABEL_32:
    v28 = v27((__int64)v3);
    goto LABEL_22;
  }
  v27 = *(__int64 (__fastcall **)(__int64))(*v3 + 24LL);
  *(_DWORD *)(a1 + 72) = *(_DWORD *)(v3[2] + 72LL);
  if ( v27 != sub_22421C0 )
    goto LABEL_32;
LABEL_21:
  v28 = *(_DWORD *)(v3[2] + 76LL);
LABEL_22:
  *(_DWORD *)(a1 + 76) = v28;
  v29 = sub_2243120(a2, (__int64)v25);
  (*(void (__fastcall **)(__int64, char *, char *, __int64))(*(_QWORD *)v29 + 88LL))(
    v29,
    off_4CDFAC0[0],
    off_4CDFAC0[0] + 36,
    a1 + 80);
  (*(void (__fastcall **)(__int64, char *, char *, __int64))(*(_QWORD *)v29 + 88LL))(
    v29,
    off_4CDFAC8[0],
    off_4CDFAC8[0] + 26,
    a1 + 224);
  v30 = v41[0];
  *(_QWORD *)(a1 + 16) = v10;
  *(_QWORD *)(a1 + 40) = v18;
  *(_QWORD *)(a1 + 56) = v25;
  *(_BYTE *)(a1 + 328) = 1;
  if ( (_UNKNOWN *)(v30 - 24) != &unk_4FD67E0 )
  {
    if ( &_pthread_key_create )
    {
      v36 = _InterlockedExchangeAdd((volatile signed __int32 *)(v30 - 8), 0xFFFFFFFF);
    }
    else
    {
      v36 = *(_DWORD *)(v30 - 8);
      *(_DWORD *)(v30 - 8) = v36 - 1;
    }
    if ( v36 <= 0 )
      j_j___libc_free_0_2(v30 - 24);
  }
  v31 = (unsigned __int64)(v40 - 3);
  if ( v40 - 3 != (_QWORD *)&unk_4FD67E0 )
  {
    if ( &_pthread_key_create )
    {
      v37 = _InterlockedExchangeAdd((volatile signed __int32 *)v40 - 2, 0xFFFFFFFF);
    }
    else
    {
      v37 = *((_DWORD *)v40 - 2);
      *((_DWORD *)v40 - 2) = v37 - 1;
    }
    if ( v37 <= 0 )
      j_j___libc_free_0_2(v31);
  }
  v32 = (unsigned __int64)(v39 - 3);
  if ( v39 - 3 != (_QWORD *)&unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v35 = _InterlockedExchangeAdd((volatile signed __int32 *)v39 - 2, 0xFFFFFFFF);
    }
    else
    {
      v35 = *((_DWORD *)v39 - 2);
      *((_DWORD *)v39 - 2) = v35 - 1;
    }
    if ( v35 <= 0 )
      j_j___libc_free_0_1(v32);
  }
}
