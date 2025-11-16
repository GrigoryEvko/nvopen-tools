// Function: sub_2F0F4A0
// Address: 0x2f0f4a0
//
void __fastcall sub_2F0F4A0(__int64 a1, __int64 a2)
{
  char v2; // al
  _BOOL8 v3; // rcx
  char v4; // al
  _BOOL8 v5; // rcx
  char v6; // al
  _BOOL8 v7; // rcx
  char v8; // al
  __int64 v9; // rcx
  char v10; // al
  _BOOL8 v11; // rcx
  char v12; // al
  __int64 v13; // rcx
  unsigned __int8 (__fastcall *v14)(__int64, char *, _BOOL8); // r14
  char v15; // al
  _BOOL8 v16; // rdx
  unsigned __int8 (__fastcall *v17)(__int64, const char *, _BOOL8); // r14
  char v18; // al
  _BOOL8 v19; // rdx
  char v20; // [rsp+Fh] [rbp-61h] BYREF
  _QWORD *v21; // [rsp+10h] [rbp-60h] BYREF
  __int64 v22; // [rsp+18h] [rbp-58h]
  _BYTE v23[16]; // [rsp+20h] [rbp-50h] BYREF
  __int128 v24; // [rsp+30h] [rbp-40h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, char *, _QWORD **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "id",
         1,
         0,
         &v20,
         &v21) )
  {
    sub_2F08170(a1, a2);
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 128LL))(a1, v21);
  }
  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v3 = 0;
  if ( v2 )
    v3 = *(_DWORD *)(a2 + 24) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, char *, _QWORD **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "type",
         0,
         v3,
         &v20,
         &v21) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v14 = *(unsigned __int8 (__fastcall **)(__int64, char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v15 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v16 = 0;
    if ( v15 )
      v16 = *(_DWORD *)(a2 + 24) == 0;
    if ( v14(a1, "default", v16) )
      *(_DWORD *)(a2 + 24) = 0;
    v17 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v18 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v19 = 0;
    if ( v18 )
      v19 = *(_DWORD *)(a2 + 24) == 1;
    if ( v17(a1, "spill-slot", v19) )
      *(_DWORD *)(a2 + 24) = 1;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 128LL))(a1, v21);
  }
  else if ( v20 )
  {
    *(_DWORD *)(a2 + 24) = 0;
  }
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v5 = 0;
  if ( v4 )
    v5 = *(_QWORD *)(a2 + 32) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, char *, _QWORD **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "offset",
         0,
         v5,
         &v20,
         &v21) )
  {
    sub_2F07F90(a1, (_QWORD *)(a2 + 32));
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 128LL))(a1, v21);
  }
  else if ( v20 )
  {
    *(_QWORD *)(a2 + 32) = 0;
  }
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v7 = 0;
  if ( v6 )
    v7 = *(_QWORD *)(a2 + 40) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, char *, _QWORD **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "size",
         0,
         v7,
         &v20,
         &v21) )
  {
    sub_2F07BD0(a1, (_QWORD *)(a2 + 40));
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 128LL))(a1, v21);
  }
  else if ( v20 )
  {
    *(_QWORD *)(a2 + 40) = 0;
  }
  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v9 = 0;
  if ( v8 )
    v9 = *(_BYTE *)(a2 + 49) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, __int64, char *, _QWORD **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "alignment",
         0,
         v9,
         &v20,
         &v21) )
  {
    sub_2F085F0(a1, (_BYTE *)(a2 + 48));
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 128LL))(a1, v21);
  }
  else if ( v20 )
  {
    *(_BYTE *)(a2 + 49) = 0;
  }
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v11 = 0;
  if ( v10 )
    v11 = *(_DWORD *)(a2 + 52) == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "stack-id",
         0,
         v11,
         &v20,
         &v21) )
  {
    sub_2F07700(a1, (_DWORD *)(a2 + 52));
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 128LL))(a1, v21);
  }
  else if ( v20 )
  {
    *(_DWORD *)(a2 + 52) = 0;
  }
  if ( *(_DWORD *)(a2 + 24) != 1 )
  {
    LOBYTE(v21) = 0;
    sub_2F07B20(a1, (__int64)"isImmutable", (_BYTE *)(a2 + 56), &v21, 0);
    LOBYTE(v21) = 0;
    sub_2F07B20(a1, (__int64)"isAliased", (_BYTE *)(a2 + 57), &v21, 0);
  }
  v21 = v23;
  v22 = 0;
  v23[0] = 0;
  v24 = 0;
  sub_2F0ECF0(a1, (__int64)"callee-saved-register", a2 + 64, (__int64)&v21, 0);
  if ( v21 != (_QWORD *)v23 )
    j_j___libc_free_0((unsigned __int64)v21);
  v12 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v13 = 0;
  if ( v12 )
    v13 = *(unsigned __int8 *)(a2 + 112);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, char *, _QWORD **))(*(_QWORD *)a1 + 120LL))(
         a1,
         "callee-saved-restored",
         0,
         v13,
         &v20,
         &v21) )
  {
    sub_2F07940(a1, (_BYTE *)(a2 + 112));
    (*(void (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)a1 + 128LL))(a1, v21);
  }
  else if ( v20 )
  {
    *(_BYTE *)(a2 + 112) = 1;
  }
  v21 = v23;
  v22 = 0;
  v23[0] = 0;
  v24 = 0;
  sub_2F0ECF0(a1, (__int64)"debug-info-variable", a2 + 120, (__int64)&v21, 0);
  if ( v21 != (_QWORD *)v23 )
    j_j___libc_free_0((unsigned __int64)v21);
  v21 = v23;
  v22 = 0;
  v23[0] = 0;
  v24 = 0;
  sub_2F0ECF0(a1, (__int64)"debug-info-expression", a2 + 168, (__int64)&v21, 0);
  if ( v21 != (_QWORD *)v23 )
    j_j___libc_free_0((unsigned __int64)v21);
  v21 = v23;
  v22 = 0;
  v23[0] = 0;
  v24 = 0;
  sub_2F0ECF0(a1, (__int64)"debug-info-location", a2 + 216, (__int64)&v21, 0);
  if ( v21 != (_QWORD *)v23 )
    j_j___libc_free_0((unsigned __int64)v21);
}
