// Function: sub_39D22A0
// Address: 0x39d22a0
//
void __fastcall sub_39D22A0(__int64 a1, __int64 a2)
{
  char v3; // al
  _BOOL8 v4; // rcx
  char v5; // al
  _BOOL8 v6; // rcx
  char v7; // al
  _BOOL8 v8; // rcx
  char v9; // al
  _BOOL8 v10; // rcx
  char v11; // al
  __int64 v12; // rcx
  __int64 v13; // rax
  char v14; // al
  _BOOL8 v15; // rcx
  size_t v16; // rdx
  void **p_s2; // rsi
  __int64 v18; // rax
  char v19; // al
  _BOOL8 v20; // rcx
  size_t v21; // rdx
  unsigned __int8 (__fastcall *v22)(__int64, char *, _BOOL8); // r14
  char v23; // al
  _BOOL8 v24; // rdx
  unsigned __int8 (__fastcall *v25)(__int64, const char *, _BOOL8); // r14
  char v26; // al
  _BOOL8 v27; // rdx
  unsigned __int64 *v28; // [rsp+0h] [rbp-80h]
  unsigned __int64 *v29; // [rsp+0h] [rbp-80h]
  char v30; // [rsp+17h] [rbp-69h] BYREF
  void **v31; // [rsp+18h] [rbp-68h] BYREF
  void *s2; // [rsp+20h] [rbp-60h] BYREF
  __int64 v33; // [rsp+28h] [rbp-58h]
  _QWORD v34[2]; // [rsp+30h] [rbp-50h] BYREF
  __m128i v35[4]; // [rsp+40h] [rbp-40h] BYREF

  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, void ***, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "id",
         1,
         0,
         &v31,
         &s2) )
  {
    sub_39D0990(a1, a2);
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v4 = 0;
  if ( v3 )
    v4 = *(_DWORD *)(a2 + 24) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, void ***, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "type",
         0,
         v4,
         &v31,
         &s2) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v22 = *(unsigned __int8 (__fastcall **)(__int64, char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v23 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v24 = 0;
    if ( v23 )
      v24 = *(_DWORD *)(a2 + 24) == 0;
    if ( v22(a1, "default", v24) )
      *(_DWORD *)(a2 + 24) = 0;
    v25 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v26 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v27 = 0;
    if ( v26 )
      v27 = *(_DWORD *)(a2 + 24) == 1;
    if ( v25(a1, "spill-slot", v27) )
      *(_DWORD *)(a2 + 24) = 1;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
  else if ( (_BYTE)v31 )
  {
    *(_DWORD *)(a2 + 24) = 0;
  }
  v5 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v6 = 0;
  if ( v5 )
    v6 = *(_QWORD *)(a2 + 32) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, void ***, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "offset",
         0,
         v6,
         &v31,
         &s2) )
  {
    sub_39D0460(a1, (_QWORD *)(a2 + 32));
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
  else if ( (_BYTE)v31 )
  {
    *(_QWORD *)(a2 + 32) = 0;
  }
  v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v8 = 0;
  if ( v7 )
    v8 = *(_QWORD *)(a2 + 40) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, void ***, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "size",
         0,
         v8,
         &v31,
         &s2) )
  {
    sub_39D05E0(a1, (_QWORD *)(a2 + 40));
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
  else if ( (_BYTE)v31 )
  {
    *(_QWORD *)(a2 + 40) = 0;
  }
  v9 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v10 = 0;
  if ( v9 )
    v10 = *(_DWORD *)(a2 + 48) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, void ***, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "alignment",
         0,
         v10,
         &v31,
         &s2) )
  {
    sub_39D02E0(a1, (unsigned int *)(a2 + 48));
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
  else if ( (_BYTE)v31 )
  {
    *(_DWORD *)(a2 + 48) = 0;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, void ***, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "stack-id",
         0,
         0,
         &v31,
         &s2) )
  {
    sub_39D0160(a1, (unsigned __int8 *)(a2 + 52));
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
  if ( *(_DWORD *)(a2 + 24) != 1 )
  {
    LOBYTE(s2) = 0;
    sub_39D08E0(a1, (__int64)"isImmutable", (_BYTE *)(a2 + 53), &s2, 0);
    LOBYTE(s2) = 0;
    sub_39D08E0(a1, (__int64)"isAliased", (_BYTE *)(a2 + 54), &s2, 0);
  }
  s2 = v34;
  v33 = 0;
  LOBYTE(v34[0]) = 0;
  v35[0] = 0;
  sub_39D1940(a1, (__int64)"callee-saved-register", a2 + 56, (__int64)&s2, 0);
  if ( s2 != v34 )
    j_j___libc_free_0((unsigned __int64)s2);
  v11 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v12 = 0;
  if ( v11 )
    v12 = *(unsigned __int8 *)(a2 + 104);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, void ***, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "callee-saved-restored",
         0,
         v12,
         &v31,
         &s2) )
  {
    sub_39D0760(a1, (_BYTE *)(a2 + 104));
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
  else if ( (_BYTE)v31 )
  {
    *(_BYTE *)(a2 + 104) = 1;
  }
  s2 = v34;
  v33 = 0;
  LOBYTE(v34[0]) = 0;
  v35[0] = 0;
  sub_39D1940(a1, (__int64)"debug-info-variable", a2 + 112, (__int64)&s2, 0);
  if ( s2 != v34 )
    j_j___libc_free_0((unsigned __int64)s2);
  s2 = v34;
  v28 = (unsigned __int64 *)(a2 + 160);
  v13 = *(_QWORD *)a1;
  v33 = 0;
  LOBYTE(v34[0]) = 0;
  v35[0] = 0;
  v14 = (*(__int64 (__fastcall **)(__int64))(v13 + 16))(a1);
  v15 = 0;
  if ( v14 )
  {
    v16 = *(_QWORD *)(a2 + 168);
    if ( v16 == v33 )
    {
      v15 = 1;
      if ( v16 )
        v15 = memcmp(*(const void **)(a2 + 160), s2, v16) == 0;
    }
  }
  p_s2 = (void **)"debug-info-expression";
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, void ***))(*(_QWORD *)a1 + 120LL))(
         a1,
         "debug-info-expression",
         0,
         v15,
         &v30,
         &v31) )
  {
    sub_39D16D0(a1, (__int64)v28);
    p_s2 = v31;
    (*(void (__fastcall **)(__int64, void **))(*(_QWORD *)a1 + 128LL))(a1, v31);
  }
  else if ( v30 )
  {
    p_s2 = &s2;
    sub_2240AE0(v28, (unsigned __int64 *)&s2);
    *(__m128i *)(a2 + 192) = _mm_loadu_si128(v35);
  }
  if ( s2 != v34 )
  {
    p_s2 = (void **)(v34[0] + 1LL);
    j_j___libc_free_0((unsigned __int64)s2);
  }
  s2 = v34;
  v29 = (unsigned __int64 *)(a2 + 208);
  v18 = *(_QWORD *)a1;
  v33 = 0;
  LOBYTE(v34[0]) = 0;
  v35[0] = 0;
  v19 = (*(__int64 (__fastcall **)(__int64, void **))(v18 + 16))(a1, p_s2);
  v20 = 0;
  if ( v19 )
  {
    v21 = *(_QWORD *)(a2 + 216);
    if ( v21 == v33 )
    {
      v20 = 1;
      if ( v21 )
        v20 = memcmp(*(const void **)(a2 + 208), s2, v21) == 0;
    }
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, void ***))(*(_QWORD *)a1 + 120LL))(
         a1,
         "debug-info-location",
         0,
         v20,
         &v30,
         &v31) )
  {
    sub_39D16D0(a1, (__int64)v29);
    (*(void (__fastcall **)(__int64, void **))(*(_QWORD *)a1 + 128LL))(a1, v31);
  }
  else if ( v30 )
  {
    sub_2240AE0(v29, (unsigned __int64 *)&s2);
    *(__m128i *)(a2 + 240) = _mm_loadu_si128(v35);
  }
  if ( s2 != v34 )
    j_j___libc_free_0((unsigned __int64)s2);
}
