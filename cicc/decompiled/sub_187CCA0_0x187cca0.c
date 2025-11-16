// Function: sub_187CCA0
// Address: 0x187cca0
//
__int64 __fastcall sub_187CCA0(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int8 (__fastcall *v3)(__int64, const char *, _BOOL8); // r14
  char v4; // al
  _BOOL8 v5; // rdx
  unsigned __int8 (__fastcall *v6)(__int64, const char *, _BOOL8); // r14
  char v7; // al
  _BOOL8 v8; // rdx
  unsigned __int8 (__fastcall *v9)(__int64, char *, _BOOL8); // r14
  char v10; // al
  _BOOL8 v11; // rdx
  unsigned __int8 (__fastcall *v12)(__int64, const char *, _BOOL8); // r14
  char v13; // al
  _BOOL8 v14; // rdx
  unsigned __int8 (__fastcall *v15)(__int64, const char *, _BOOL8); // r14
  char v16; // al
  _BOOL8 v17; // rdx
  __int64 v18; // rax
  __int64 v19; // rax
  const char *v20; // rax
  __int64 v21; // rdx
  void (__fastcall *v22)(__int64, const char ***); // rax
  __int64 v23; // rax
  char v24; // [rsp+17h] [rbp-99h] BYREF
  __int64 v25; // [rsp+18h] [rbp-98h] BYREF
  __int64 v26; // [rsp+20h] [rbp-90h] BYREF
  __int64 v27; // [rsp+28h] [rbp-88h]
  const char *v28; // [rsp+30h] [rbp-80h] BYREF
  __int64 v29; // [rsp+38h] [rbp-78h]
  _QWORD v30[2]; // [rsp+40h] [rbp-70h] BYREF
  const char **v31; // [rsp+50h] [rbp-60h] BYREF
  __int64 v32; // [rsp+58h] [rbp-58h]
  __int64 v33; // [rsp+60h] [rbp-50h]
  __int64 v34; // [rsp+68h] [rbp-48h]
  int v35; // [rsp+70h] [rbp-40h]
  __int64 *v36; // [rsp+78h] [rbp-38h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, const char **, const char ***))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Kind",
         0,
         0,
         &v28,
         &v31) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v3 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v5 = 0;
    if ( v4 )
      v5 = *(_DWORD *)a2 == 0;
    if ( v3(a1, "Unsat", v5) )
      *(_DWORD *)a2 = 0;
    v6 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v8 = 0;
    if ( v7 )
      v8 = *(_DWORD *)a2 == 1;
    if ( v6(a1, "ByteArray", v8) )
      *(_DWORD *)a2 = 1;
    v9 = *(unsigned __int8 (__fastcall **)(__int64, char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v11 = 0;
    if ( v10 )
      v11 = *(_DWORD *)a2 == 2;
    if ( v9(a1, "Inline", v11) )
      *(_DWORD *)a2 = 2;
    v12 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v13 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v14 = 0;
    if ( v13 )
      v14 = *(_DWORD *)a2 == 3;
    if ( v12(a1, "Single", v14) )
      *(_DWORD *)a2 = 3;
    v15 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v16 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v17 = 0;
    if ( v16 )
      v17 = *(_DWORD *)a2 == 4;
    if ( v15(a1, "AllOnes", v17) )
      *(_DWORD *)a2 = 4;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, const char **))(*(_QWORD *)a1 + 128LL))(a1, v31);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, const char **, const char ***))(*(_QWORD *)a1 + 120LL))(
         a1,
         "SizeM1BitWidth",
         0,
         0,
         &v28,
         &v31) )
  {
    sub_1879C60(a1, (unsigned int *)(a2 + 4));
    (*(void (__fastcall **)(__int64, const char **))(*(_QWORD *)a1 + 128LL))(a1, v31);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, const char **, const char ***))(*(_QWORD *)a1 + 120LL))(
         a1,
         "AlignLog2",
         0,
         0,
         &v28,
         &v31) )
  {
    sub_1879F60(a1, (_QWORD *)(a2 + 8));
    (*(void (__fastcall **)(__int64, const char **))(*(_QWORD *)a1 + 128LL))(a1, v31);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, const char **, const char ***))(*(_QWORD *)a1 + 120LL))(
         a1,
         "SizeM1",
         0,
         0,
         &v28,
         &v31) )
  {
    sub_1879F60(a1, (_QWORD *)(a2 + 16));
    (*(void (__fastcall **)(__int64, const char **))(*(_QWORD *)a1 + 128LL))(a1, v31);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "BitMask",
         0,
         0,
         &v24,
         &v25) )
  {
    if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    {
      LOBYTE(v30[0]) = 0;
      v28 = (const char *)v30;
      v29 = 0;
      v35 = 1;
      v34 = 0;
      v33 = 0;
      v32 = 0;
      v31 = (const char **)&unk_49EFBE0;
      v36 = (__int64 *)&v28;
      v23 = sub_16E4080(a1);
      sub_16E5960((unsigned __int8 *)(a2 + 24), v23, (__int64)&v31);
      if ( v34 != v32 )
        sub_16E7BA0((__int64 *)&v31);
      v26 = *v36;
      v27 = v36[1];
      (*(void (__fastcall **)(__int64, __int64 *, _QWORD))(*(_QWORD *)a1 + 216LL))(a1, &v26, 0);
      sub_16E7BC0((__int64 *)&v31);
      if ( v28 != (const char *)v30 )
        j_j___libc_free_0(v28, v30[0] + 1LL);
    }
    else
    {
      v18 = *(_QWORD *)a1;
      v26 = 0;
      v27 = 0;
      (*(void (__fastcall **)(__int64, __int64 *, _QWORD))(v18 + 216))(a1, &v26, 0);
      v19 = sub_16E4080(a1);
      v20 = sub_16E5970(v26, v27, v19, (_BYTE *)(a2 + 24));
      v29 = v21;
      v28 = v20;
      if ( v21 )
      {
        v22 = *(void (__fastcall **)(__int64, const char ***))(*(_QWORD *)a1 + 232LL);
        LOWORD(v33) = 261;
        v31 = &v28;
        v22(a1, &v31);
      }
    }
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v25);
  }
  result = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, const char **, const char ***))(*(_QWORD *)a1 + 120LL))(
             a1,
             "InlineBits",
             0,
             0,
             &v28,
             &v31);
  if ( (_BYTE)result )
  {
    sub_1879F60(a1, (_QWORD *)(a2 + 32));
    return (*(__int64 (__fastcall **)(__int64, const char **))(*(_QWORD *)a1 + 128LL))(a1, v31);
  }
  return result;
}
