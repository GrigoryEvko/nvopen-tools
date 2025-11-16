// Function: sub_39D1C00
// Address: 0x39d1c00
//
void __fastcall sub_39D1C00(__int64 a1, __int64 a2)
{
  char v2; // al
  _BOOL8 v3; // rcx
  char v4; // al
  _BOOL8 v5; // rcx
  char v6; // al
  _BOOL8 v7; // rcx
  char v8; // al
  __int64 v9; // rcx
  __int64 v10; // rax
  unsigned __int8 v11; // dl
  __int64 v12; // rax
  char v13; // al
  _BOOL8 v14; // rcx
  size_t v15; // rdx
  bool v16; // zf
  char v17; // al
  _BOOL8 v18; // rdx
  char v19; // al
  _BOOL8 v20; // rdx
  char v21; // al
  _BOOL8 v22; // rdx
  unsigned __int8 v23; // [rsp+8h] [rbp-78h]
  unsigned __int64 *v24; // [rsp+8h] [rbp-78h]
  unsigned __int8 (__fastcall *v25)(__int64, char *, _BOOL8); // [rsp+8h] [rbp-78h]
  unsigned __int8 (__fastcall *v26)(__int64, const char *, _BOOL8); // [rsp+8h] [rbp-78h]
  unsigned __int8 (__fastcall *v27)(__int64, const char *, _BOOL8); // [rsp+8h] [rbp-78h]
  char v28; // [rsp+17h] [rbp-69h] BYREF
  __int64 v29; // [rsp+18h] [rbp-68h] BYREF
  void *s2; // [rsp+20h] [rbp-60h] BYREF
  __int64 v31; // [rsp+28h] [rbp-58h]
  _BYTE v32[16]; // [rsp+30h] [rbp-50h] BYREF
  __m128i v33[4]; // [rsp+40h] [rbp-40h] BYREF

  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, __int64 *, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "id",
         1,
         0,
         &v29,
         &s2) )
  {
    sub_39D0990(a1, a2);
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
  s2 = v32;
  v31 = 0;
  v32[0] = 0;
  v33[0] = 0;
  sub_39D1940(a1, (__int64)"name", a2 + 24, (__int64)&s2, 0);
  if ( s2 != v32 )
    j_j___libc_free_0((unsigned __int64)s2);
  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v3 = 0;
  if ( v2 )
    v3 = *(_DWORD *)(a2 + 72) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, __int64 *, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "type",
         0,
         v3,
         &v29,
         &s2) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v25 = *(unsigned __int8 (__fastcall **)(__int64, char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v17 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v18 = 0;
    if ( v17 )
      v18 = *(_DWORD *)(a2 + 72) == 0;
    if ( v25(a1, "default", v18) )
      *(_DWORD *)(a2 + 72) = 0;
    v26 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v19 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v20 = 0;
    if ( v19 )
      v20 = *(_DWORD *)(a2 + 72) == 1;
    if ( v26(a1, "spill-slot", v20) )
      *(_DWORD *)(a2 + 72) = 1;
    v27 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v21 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v22 = 0;
    if ( v21 )
      v22 = *(_DWORD *)(a2 + 72) == 2;
    if ( v27(a1, "variable-sized", v22) )
      *(_DWORD *)(a2 + 72) = 2;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
  else if ( (_BYTE)v29 )
  {
    *(_DWORD *)(a2 + 72) = 0;
  }
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v5 = 0;
  if ( v4 )
    v5 = *(_QWORD *)(a2 + 80) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, __int64 *, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "offset",
         0,
         v5,
         &v29,
         &s2) )
  {
    sub_39D0460(a1, (_QWORD *)(a2 + 80));
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
  else if ( (_BYTE)v29 )
  {
    v16 = *(_DWORD *)(a2 + 72) == 2;
    *(_QWORD *)(a2 + 80) = 0;
    if ( v16 )
      goto LABEL_15;
    goto LABEL_52;
  }
  if ( *(_DWORD *)(a2 + 72) == 2 )
    goto LABEL_15;
LABEL_52:
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, __int64 *, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "size",
         1,
         0,
         &v29,
         &s2) )
  {
    sub_39D05E0(a1, (_QWORD *)(a2 + 88));
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
LABEL_15:
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v7 = 0;
  if ( v6 )
    v7 = *(_DWORD *)(a2 + 96) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, __int64 *, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "alignment",
         0,
         v7,
         &v29,
         &s2) )
  {
    sub_39D02E0(a1, (unsigned int *)(a2 + 96));
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
  else if ( (_BYTE)v29 )
  {
    *(_DWORD *)(a2 + 96) = 0;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __int64 *, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "stack-id",
         0,
         0,
         &v29,
         &s2) )
  {
    sub_39D0160(a1, (unsigned __int8 *)(a2 + 100));
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
  s2 = v32;
  v31 = 0;
  v32[0] = 0;
  v33[0] = 0;
  sub_39D1940(a1, (__int64)"callee-saved-register", a2 + 104, (__int64)&s2, 0);
  if ( s2 != v32 )
    j_j___libc_free_0((unsigned __int64)s2);
  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v9 = 0;
  if ( v8 )
    v9 = *(unsigned __int8 *)(a2 + 152);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, __int64 *, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "callee-saved-restored",
         0,
         v9,
         &v29,
         &s2) )
  {
    sub_39D0760(a1, (_BYTE *)(a2 + 152));
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
  else if ( (_BYTE)v29 )
  {
    *(_BYTE *)(a2 + 152) = 1;
  }
  v10 = *(_QWORD *)a1;
  LOBYTE(v29) = 1;
  v11 = (*(__int64 (__fastcall **)(__int64))(v10 + 16))(a1);
  if ( v11 )
    v11 = *(_BYTE *)(a2 + 168) ^ 1;
  v23 = v11;
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    if ( !*(_BYTE *)(a2 + 168) )
      goto LABEL_39;
  }
  else if ( !*(_BYTE *)(a2 + 168) )
  {
    *(_QWORD *)(a2 + 160) = 0;
    *(_BYTE *)(a2 + 168) = 1;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __int64 *, void **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "local-offset",
         0,
         v23,
         &v29,
         &s2) )
  {
    sub_39D0460(a1, (_QWORD *)(a2 + 160));
    (*(void (__fastcall **)(__int64, void *))(*(_QWORD *)a1 + 128LL))(a1, s2);
  }
  else if ( (_BYTE)v29 && *(_BYTE *)(a2 + 168) )
  {
    *(_BYTE *)(a2 + 168) = 0;
  }
LABEL_39:
  s2 = v32;
  v24 = (unsigned __int64 *)(a2 + 176);
  v12 = *(_QWORD *)a1;
  v31 = 0;
  v32[0] = 0;
  v33[0] = 0;
  v13 = (*(__int64 (__fastcall **)(__int64))(v12 + 16))(a1);
  v14 = 0;
  if ( v13 )
  {
    v15 = *(_QWORD *)(a2 + 184);
    if ( v15 == v31 )
    {
      v14 = 1;
      if ( v15 )
        v14 = memcmp(*(const void **)(a2 + 176), s2, v15) == 0;
    }
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "debug-info-variable",
         0,
         v14,
         &v28,
         &v29) )
  {
    sub_39D16D0(a1, (__int64)v24);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v29);
  }
  else if ( v28 )
  {
    sub_2240AE0(v24, (unsigned __int64 *)&s2);
    *(__m128i *)(a2 + 208) = _mm_loadu_si128(v33);
  }
  if ( s2 != v32 )
    j_j___libc_free_0((unsigned __int64)s2);
  s2 = v32;
  v31 = 0;
  v32[0] = 0;
  v33[0] = 0;
  sub_39D1940(a1, (__int64)"debug-info-expression", a2 + 224, (__int64)&s2, 0);
  if ( s2 != v32 )
    j_j___libc_free_0((unsigned __int64)s2);
  s2 = v32;
  v31 = 0;
  v32[0] = 0;
  v33[0] = 0;
  sub_39D1940(a1, (__int64)"debug-info-location", a2 + 272, (__int64)&s2, 0);
  if ( s2 != v32 )
    j_j___libc_free_0((unsigned __int64)s2);
}
