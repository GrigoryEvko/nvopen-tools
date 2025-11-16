// Function: sub_2F0EDE0
// Address: 0x2f0ede0
//
void __fastcall sub_2F0EDE0(__int64 a1, __int64 a2)
{
  char v2; // al
  _BOOL8 v3; // rcx
  char v4; // al
  _BOOL8 v5; // rcx
  char v6; // al
  __int64 v7; // rcx
  char v8; // al
  _BOOL8 v9; // rcx
  __int64 v10; // rax
  unsigned __int8 v11; // r15
  __int64 v12; // rax
  unsigned __int64 v13; // r15
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  bool v16; // zf
  char v17; // al
  _BOOL8 v18; // rdx
  char v19; // al
  _BOOL8 v20; // rdx
  char v21; // al
  _BOOL8 v22; // rdx
  unsigned __int8 (__fastcall *v23)(__int64, char *, _BOOL8); // [rsp+8h] [rbp-88h]
  unsigned __int8 (__fastcall *v24)(__int64, const char *, _BOOL8); // [rsp+8h] [rbp-88h]
  unsigned __int8 (__fastcall *v25)(__int64, const char *, _BOOL8); // [rsp+8h] [rbp-88h]
  char v26; // [rsp+17h] [rbp-79h] BYREF
  __int64 v27; // [rsp+18h] [rbp-78h] BYREF
  __m128i v28; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v29; // [rsp+30h] [rbp-60h] BYREF
  unsigned __int64 v30; // [rsp+38h] [rbp-58h]
  _BYTE v31[16]; // [rsp+40h] [rbp-50h] BYREF
  __int128 v32; // [rsp+50h] [rbp-40h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, __m128i *, _QWORD **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "id",
         1,
         0,
         &v28,
         &v29) )
  {
    sub_2F08170(a1, a2);
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 128LL))(a1, v29);
  }
  v29 = v31;
  v30 = 0;
  v31[0] = 0;
  v32 = 0;
  sub_2F0ECF0(a1, (__int64)"name", a2 + 24, (__int64)&v29, 0);
  if ( v29 != (_QWORD *)v31 )
    j_j___libc_free_0((unsigned __int64)v29);
  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v3 = 0;
  if ( v2 )
    v3 = *(_DWORD *)(a2 + 72) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, __m128i *, _QWORD **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "type",
         0,
         v3,
         &v28,
         &v29) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v23 = *(unsigned __int8 (__fastcall **)(__int64, char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v17 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v18 = 0;
    if ( v17 )
      v18 = *(_DWORD *)(a2 + 72) == 0;
    if ( v23(a1, "default", v18) )
      *(_DWORD *)(a2 + 72) = 0;
    v24 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v19 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v20 = 0;
    if ( v19 )
      v20 = *(_DWORD *)(a2 + 72) == 1;
    if ( v24(a1, "spill-slot", v20) )
      *(_DWORD *)(a2 + 72) = 1;
    v25 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v21 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v22 = 0;
    if ( v21 )
      v22 = *(_DWORD *)(a2 + 72) == 2;
    if ( v25(a1, "variable-sized", v22) )
      *(_DWORD *)(a2 + 72) = 2;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 128LL))(a1, v29);
  }
  else if ( v28.m128i_i8[0] )
  {
    *(_DWORD *)(a2 + 72) = 0;
  }
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v5 = 0;
  if ( v4 )
    v5 = *(_QWORD *)(a2 + 80) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, __m128i *, _QWORD **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "offset",
         0,
         v5,
         &v28,
         &v29) )
  {
    sub_2F07F90(a1, (_QWORD *)(a2 + 80));
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 128LL))(a1, v29);
  }
  else if ( v28.m128i_i8[0] )
  {
    v16 = *(_DWORD *)(a2 + 72) == 2;
    *(_QWORD *)(a2 + 80) = 0;
    if ( v16 )
      goto LABEL_15;
    goto LABEL_54;
  }
  if ( *(_DWORD *)(a2 + 72) == 2 )
    goto LABEL_15;
LABEL_54:
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, __m128i *, _QWORD **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "size",
         1,
         0,
         &v28,
         &v29) )
  {
    sub_2F07BD0(a1, (_QWORD *)(a2 + 88));
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 128LL))(a1, v29);
  }
LABEL_15:
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v7 = 0;
  if ( v6 )
    v7 = *(_BYTE *)(a2 + 97) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, __int64, __m128i *, _QWORD **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "alignment",
         0,
         v7,
         &v28,
         &v29) )
  {
    sub_2F085F0(a1, (_BYTE *)(a2 + 96));
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 128LL))(a1, v29);
  }
  else if ( v28.m128i_i8[0] )
  {
    *(_BYTE *)(a2 + 97) = 0;
  }
  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v9 = 0;
  if ( v8 )
    v9 = *(_DWORD *)(a2 + 100) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, __m128i *, _QWORD **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "stack-id",
         0,
         v9,
         &v28,
         &v29) )
  {
    sub_2F07700(a1, (_DWORD *)(a2 + 100));
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 128LL))(a1, v29);
  }
  else if ( v28.m128i_i8[0] )
  {
    *(_DWORD *)(a2 + 100) = 0;
  }
  v29 = v31;
  v30 = 0;
  v31[0] = 0;
  v32 = 0;
  sub_2F0ECF0(a1, (__int64)"callee-saved-register", a2 + 104, (__int64)&v29, 0);
  if ( v29 != (_QWORD *)v31 )
    j_j___libc_free_0((unsigned __int64)v29);
  LOBYTE(v29) = 1;
  sub_2F07B20(a1, (__int64)"callee-saved-restored", (_BYTE *)(a2 + 152), &v29, 0);
  v10 = *(_QWORD *)a1;
  v26 = 1;
  v28.m128i_i64[1] = 0;
  v11 = (*(__int64 (__fastcall **)(__int64))(v10 + 16))(a1);
  if ( v11 )
    v11 = *(_BYTE *)(a2 + 168) ^ 1;
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
  {
    if ( !*(_BYTE *)(a2 + 168) )
      goto LABEL_31;
  }
  else if ( !*(_BYTE *)(a2 + 168) )
  {
    *(_QWORD *)(a2 + 160) = 0;
    *(_BYTE *)(a2 + 168) = 1;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "local-offset",
          0,
          v11,
          &v26,
          &v27) )
  {
LABEL_31:
    if ( v26 )
      *(__m128i *)(a2 + 160) = _mm_loadu_si128(&v28);
    goto LABEL_33;
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    goto LABEL_56;
  v12 = sub_CB1000(a1);
  if ( *(_DWORD *)(v12 + 32) != 1 )
    goto LABEL_56;
  v13 = *(_QWORD *)(v12 + 80);
  v29 = *(_QWORD **)(v12 + 72);
  v30 = v13;
  v14 = sub_C93710(&v29, 32, 0xFFFFFFFFFFFFFFFFLL) + 1;
  if ( v14 > v30 )
    v14 = v30;
  v15 = v30 - v13 + v14;
  if ( v15 > v30 )
    v15 = v30;
  if ( v15 == 6 && *(_DWORD *)v29 == 1852796476 && *((_WORD *)v29 + 2) == 15973 )
    *(__m128i *)(a2 + 160) = _mm_loadu_si128(&v28);
  else
LABEL_56:
    sub_2F07F90(a1, (_QWORD *)(a2 + 160));
  (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v27);
LABEL_33:
  v29 = v31;
  v30 = 0;
  v31[0] = 0;
  v32 = 0;
  sub_2F0ECF0(a1, (__int64)"debug-info-variable", a2 + 176, (__int64)&v29, 0);
  if ( v29 != (_QWORD *)v31 )
    j_j___libc_free_0((unsigned __int64)v29);
  v29 = v31;
  v30 = 0;
  v31[0] = 0;
  v32 = 0;
  sub_2F0ECF0(a1, (__int64)"debug-info-expression", a2 + 224, (__int64)&v29, 0);
  if ( v29 != (_QWORD *)v31 )
    j_j___libc_free_0((unsigned __int64)v29);
  v29 = v31;
  v30 = 0;
  v31[0] = 0;
  v32 = 0;
  sub_2F0ECF0(a1, (__int64)"debug-info-location", a2 + 272, (__int64)&v29, 0);
  if ( v29 != (_QWORD *)v31 )
    j_j___libc_free_0((unsigned __int64)v29);
}
