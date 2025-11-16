// Function: sub_2243C60
// Address: 0x2243c60
//
void __fastcall sub_2243C60(__int64 a1, _QWORD *a2)
{
  _QWORD *v3; // rbp
  _QWORD *v4; // rax
  __int64 (__fastcall *v5)(__int64); // rdx
  int v6; // edx
  __int64 (__fastcall *v7)(__int64); // rdx
  int v8; // edx
  __int64 (__fastcall *v9)(__int64); // rdx
  int v10; // edx
  _QWORD *(__fastcall *v11)(_QWORD *, __int64); // rax
  __int64 v12; // rax
  __int64 v13; // rsi
  _BYTE *v14; // r12
  _QWORD *v15; // rax
  unsigned __int64 v16; // rdi
  bool v17; // al
  __int64 *(__fastcall *v18)(__int64 *, __int64); // rax
  __int64 v19; // rax
  wchar_t *v20; // r14
  __int64 v21; // rsi
  _QWORD *v22; // rax
  unsigned __int64 v23; // rdi
  wchar_t *v24; // r14
  __int64 *(__fastcall *v25)(__int64 *, __int64); // rax
  __int64 v26; // rax
  wchar_t *v27; // r13
  __int64 v28; // rsi
  _QWORD *v29; // rax
  unsigned __int64 v30; // rdi
  wchar_t *v31; // r13
  __int64 *(__fastcall *v32)(__int64 *, __int64); // rax
  __int64 v33; // rax
  wchar_t *v34; // r12
  __int64 v35; // rsi
  _QWORD *v36; // rax
  unsigned __int64 v37; // rdi
  wchar_t *v38; // rsi
  __int64 (__fastcall *v39)(__int64); // rax
  int v40; // eax
  __int64 (__fastcall *v41)(__int64); // rax
  int v42; // eax
  __int64 v43; // rax
  __int64 v44; // rax
  unsigned __int64 v45; // rdi
  unsigned __int64 v46; // rdi
  unsigned __int64 v47; // rdi
  int v48; // edx
  int v49; // edx
  int v50; // edx
  int v51; // edx
  _BYTE *v52; // [rsp+8h] [rbp-90h]
  _QWORD *v54; // [rsp+40h] [rbp-58h] BYREF
  _QWORD *v55; // [rsp+48h] [rbp-50h] BYREF
  _QWORD *v56; // [rsp+50h] [rbp-48h] BYREF
  __int64 v57[8]; // [rsp+58h] [rbp-40h] BYREF

  v3 = (_QWORD *)sub_2243C10(a2, (__int64)a2);
  v4 = (_QWORD *)*v3;
  v5 = *(__int64 (__fastcall **)(__int64))(*v3 + 16LL);
  if ( v5 == sub_2242160 )
  {
    v6 = *(_DWORD *)(v3[2] + 36LL);
  }
  else
  {
    v6 = v5((__int64)v3);
    v4 = (_QWORD *)*v3;
  }
  *(_DWORD *)(a1 + 36) = v6;
  v7 = (__int64 (__fastcall *)(__int64))v4[3];
  if ( v7 == sub_2242170 )
  {
    v8 = *(_DWORD *)(v3[2] + 40LL);
  }
  else
  {
    v8 = v7((__int64)v3);
    v4 = (_QWORD *)*v3;
  }
  *(_DWORD *)(a1 + 40) = v8;
  v9 = (__int64 (__fastcall *)(__int64))v4[8];
  if ( v9 == sub_2242180 )
  {
    v10 = *(_DWORD *)(v3[2] + 96LL);
  }
  else
  {
    v10 = v9((__int64)v3);
    v4 = (_QWORD *)*v3;
  }
  v11 = (_QWORD *(__fastcall *)(_QWORD *, __int64))v4[4];
  *(_DWORD *)(a1 + 96) = v10;
  if ( v11 == sub_22427E0 )
  {
    v12 = v3[2];
    v13 = -1;
    v14 = *(_BYTE **)(v12 + 16);
    if ( v14 )
      v13 = (__int64)&v14[strlen(*(const char **)(v12 + 16))];
    v15 = sub_22424F0(v14, (_BYTE *)v13);
    v54 = v15;
  }
  else
  {
    v11(&v54, (__int64)v3);
    v15 = v54;
  }
  v16 = *(v15 - 3);
  *(_QWORD *)(a1 + 24) = v16;
  v52 = (_BYTE *)sub_2207820(v16);
  sub_2215320(&v54, v52, *(_QWORD *)(a1 + 24), 0);
  v17 = 0;
  if ( *(_QWORD *)(a1 + 24) )
    v17 = (unsigned __int8)(*v52 - 1) <= 0x7Du;
  *(_BYTE *)(a1 + 32) = v17;
  v18 = *(__int64 *(__fastcall **)(__int64 *, __int64))(*v3 + 40LL);
  if ( v18 == sub_2242BB0 )
  {
    v19 = v3[2];
    v20 = *(wchar_t **)(v19 + 48);
    if ( !v20 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v21 = (__int64)&v20[wcslen(*(const wchar_t **)(v19 + 48))];
    if ( v20 == (wchar_t *)v21 )
      v22 = &unk_4FD67F8;
    else
      v22 = (_QWORD *)sub_2242590(v20, v21);
    v55 = v22;
  }
  else
  {
    v21 = (__int64)v3;
    v18((__int64 *)&v55, (__int64)v3);
    v22 = v55;
  }
  v23 = *(v22 - 3);
  *(_QWORD *)(a1 + 56) = v23;
  if ( v23 > 0x1FFFFFFFFFFFFFFELL )
    sub_426640(v23, v21);
  v24 = (wchar_t *)sub_2207820(4 * v23);
  sub_2215FA0((__int64 *)&v55, v24, *(_QWORD *)(a1 + 56), 0);
  v25 = *(__int64 *(__fastcall **)(__int64 *, __int64))(*v3 + 48LL);
  if ( v25 == sub_22428A0 )
  {
    v26 = v3[2];
    v27 = *(wchar_t **)(v26 + 64);
    if ( !v27 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v28 = (__int64)&v27[wcslen(*(const wchar_t **)(v26 + 64))];
    if ( v27 == (wchar_t *)v28 )
      v29 = &unk_4FD67F8;
    else
      v29 = (_QWORD *)sub_2242590(v27, v28);
    v56 = v29;
  }
  else
  {
    v28 = (__int64)v3;
    v25((__int64 *)&v56, (__int64)v3);
    v29 = v56;
  }
  v30 = *(v29 - 3);
  *(_QWORD *)(a1 + 72) = v30;
  if ( v30 > 0x1FFFFFFFFFFFFFFELL )
    sub_426640(v30, v28);
  v31 = (wchar_t *)sub_2207820(4 * v30);
  sub_2215FA0((__int64 *)&v56, v31, *(_QWORD *)(a1 + 72), 0);
  v32 = *(__int64 *(__fastcall **)(__int64 *, __int64))(*v3 + 56LL);
  if ( v32 == sub_2242910 )
  {
    v33 = v3[2];
    v34 = *(wchar_t **)(v33 + 80);
    if ( !v34 )
      sub_426248((__int64)"basic_string::_S_construct null not valid");
    v35 = (__int64)&v34[wcslen(*(const wchar_t **)(v33 + 80))];
    if ( v34 == (wchar_t *)v35 )
      v36 = &unk_4FD67F8;
    else
      v36 = (_QWORD *)sub_2242590(v34, v35);
    v57[0] = (__int64)v36;
  }
  else
  {
    v35 = (__int64)v3;
    v32(v57, (__int64)v3);
    v36 = (_QWORD *)v57[0];
  }
  v37 = *(v36 - 3);
  *(_QWORD *)(a1 + 88) = v37;
  if ( v37 > 0x1FFFFFFFFFFFFFFELL )
    sub_426640(v37, v35);
  v38 = (wchar_t *)sub_2207820(4 * v37);
  sub_2215FA0(v57, v38, *(_QWORD *)(a1 + 88), 0);
  v39 = *(__int64 (__fastcall **)(__int64))(*v3 + 72LL);
  if ( v39 == sub_2242190 )
    v40 = *(_DWORD *)(v3[2] + 100LL);
  else
    v40 = v39((__int64)v3);
  *(_DWORD *)(a1 + 100) = v40;
  v41 = *(__int64 (__fastcall **)(__int64))(*v3 + 80LL);
  if ( v41 == sub_22421A0 )
    v42 = *(_DWORD *)(v3[2] + 104LL);
  else
    v42 = v41((__int64)v3);
  *(_DWORD *)(a1 + 104) = v42;
  v43 = sub_2243120(a2, (__int64)v38);
  (*(void (__fastcall **)(__int64, char *, char *, __int64))(*(_QWORD *)v43 + 88LL))(
    v43,
    off_4CDFAD0,
    off_4CDFAD0 + 11,
    a1 + 108);
  *(_QWORD *)(a1 + 48) = v24;
  *(_QWORD *)(a1 + 64) = v31;
  *(_QWORD *)(a1 + 16) = v52;
  v44 = v57[0];
  *(_QWORD *)(a1 + 80) = v38;
  *(_BYTE *)(a1 + 152) = 1;
  if ( (_UNKNOWN *)(v44 - 24) != &unk_4FD67E0 )
  {
    if ( &_pthread_key_create )
    {
      v48 = _InterlockedExchangeAdd((volatile signed __int32 *)(v44 - 8), 0xFFFFFFFF);
    }
    else
    {
      v48 = *(_DWORD *)(v44 - 8);
      *(_DWORD *)(v44 - 8) = v48 - 1;
    }
    if ( v48 <= 0 )
      j_j___libc_free_0_2(v44 - 24);
  }
  v45 = (unsigned __int64)(v56 - 3);
  if ( v56 - 3 != (_QWORD *)&unk_4FD67E0 )
  {
    if ( &_pthread_key_create )
    {
      v49 = _InterlockedExchangeAdd((volatile signed __int32 *)v56 - 2, 0xFFFFFFFF);
    }
    else
    {
      v49 = *((_DWORD *)v56 - 2);
      *((_DWORD *)v56 - 2) = v49 - 1;
    }
    if ( v49 <= 0 )
      j_j___libc_free_0_2(v45);
  }
  v46 = (unsigned __int64)(v55 - 3);
  if ( v55 - 3 != (_QWORD *)&unk_4FD67E0 )
  {
    if ( &_pthread_key_create )
    {
      v50 = _InterlockedExchangeAdd((volatile signed __int32 *)v55 - 2, 0xFFFFFFFF);
    }
    else
    {
      v50 = *((_DWORD *)v55 - 2);
      *((_DWORD *)v55 - 2) = v50 - 1;
    }
    if ( v50 <= 0 )
      j_j___libc_free_0_2(v46);
  }
  v47 = (unsigned __int64)(v54 - 3);
  if ( v54 - 3 != (_QWORD *)&unk_4FD67C0 )
  {
    if ( &_pthread_key_create )
    {
      v51 = _InterlockedExchangeAdd((volatile signed __int32 *)v54 - 2, 0xFFFFFFFF);
    }
    else
    {
      v51 = *((_DWORD *)v54 - 2);
      *((_DWORD *)v54 - 2) = v51 - 1;
    }
    if ( v51 <= 0 )
      j_j___libc_free_0_1(v47);
  }
}
