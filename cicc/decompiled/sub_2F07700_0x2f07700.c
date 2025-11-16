// Function: sub_2F07700
// Address: 0x2f07700
//
__int64 __fastcall sub_2F07700(__int64 a1, _DWORD *a2)
{
  unsigned __int8 (__fastcall *v2)(__int64, char *, _BOOL8); // r13
  char v3; // al
  _BOOL8 v4; // rdx
  unsigned __int8 (__fastcall *v5)(__int64, const char *, _BOOL8); // r13
  char v6; // al
  _BOOL8 v7; // rdx
  unsigned __int8 (__fastcall *v8)(__int64, const char *, _BOOL8); // r13
  char v9; // al
  _BOOL8 v10; // rdx
  unsigned __int8 (__fastcall *v11)(__int64, const char *, _BOOL8); // r13
  char v12; // al
  _BOOL8 v13; // rdx
  unsigned __int8 (__fastcall *v14)(__int64, char *, _BOOL8); // r13
  char v15; // al
  _BOOL8 v16; // rdx

  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
  v2 = *(unsigned __int8 (__fastcall **)(__int64, char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
  v3 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v4 = 0;
  if ( v3 )
    v4 = *a2 == 0;
  if ( v2(a1, "default", v4) )
    *a2 = 0;
  v5 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v7 = 0;
  if ( v6 )
    v7 = *a2 == 1;
  if ( v5(a1, "sgpr-spill", v7) )
    *a2 = 1;
  v8 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
  v9 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v10 = 0;
  if ( v9 )
    v10 = *a2 == 2;
  if ( v8(a1, "scalable-vector", v10) )
    *a2 = 2;
  v11 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
  v12 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v13 = 0;
  if ( v12 )
    v13 = *a2 == 3;
  if ( v11(a1, "wasm-local", v13) )
    *a2 = 3;
  v14 = *(unsigned __int8 (__fastcall **)(__int64, char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
  v15 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v16 = 0;
  if ( v15 )
    v16 = *a2 == 255;
  if ( v14(a1, "noalloc", v16) )
    *a2 = 255;
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
}
