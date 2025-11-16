// Function: sub_1C15940
// Address: 0x1c15940
//
__int64 __fastcall sub_1C15940(__int64 a1, __int64 a2)
{
  __int64 v2; // rax
  int v3; // eax
  __int64 result; // rax
  unsigned __int8 (__fastcall *v5)(__int64, const char *, _BOOL8); // r13
  char v6; // al
  _BOOL8 v7; // rdx
  unsigned __int8 (__fastcall *v8)(__int64, const char *, _BOOL8); // r13
  char v9; // al
  _BOOL8 v10; // rdx
  unsigned __int8 (__fastcall *v11)(__int64, const char *, _BOOL8); // r13
  char v12; // al
  _BOOL8 v13; // rdx
  char v14; // [rsp+17h] [rbp-49h] BYREF
  __int64 v15; // [rsp+18h] [rbp-48h] BYREF
  __int64 v16; // [rsp+20h] [rbp-40h] BYREF
  _QWORD *v17; // [rsp+28h] [rbp-38h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, __int64 *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Type",
         1,
         0,
         &v15,
         &v16) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v5 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v7 = 0;
    if ( v6 )
      v7 = *(_DWORD *)a2 == 1;
    if ( v5(a1, "NVVM_MEMORY_WINDOW_SPECIAL_REGISTER", v7) )
      *(_DWORD *)a2 = 1;
    v8 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v9 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v10 = 0;
    if ( v9 )
      v10 = *(_DWORD *)a2 == 2;
    if ( v8(a1, "NVVM_MEMORY_WINDOW_CBANK", v10) )
      *(_DWORD *)a2 = 2;
    v11 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v12 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v13 = 0;
    if ( v12 )
      v13 = *(_DWORD *)a2 == 0;
    if ( v11(a1, "NVVM_MEMORY_WINDOW_IMMEDIATE", v13) )
      *(_DWORD *)a2 = 0;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v16);
  }
  v2 = *(_QWORD *)(a2 + 8);
  v17 = (_QWORD *)(a2 + 8);
  v16 = v2;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "StartAddress",
         0,
         0,
         &v14,
         &v15) )
  {
    sub_1C141E0(a1, &v16);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v15);
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *v17 = v16;
  LODWORD(v16) = *(_DWORD *)(a2 + 8);
  v17 = (_QWORD *)(a2 + 8);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "CBank",
         0,
         0,
         &v14,
         &v15) )
  {
    sub_1C14060(a1, (int *)&v16);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v15);
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_DWORD *)v17 = v16;
  LODWORD(v16) = *(_DWORD *)(a2 + 12);
  v17 = (_QWORD *)(a2 + 12);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "CBankOfstLow",
         0,
         0,
         &v14,
         &v15) )
  {
    sub_1C14060(a1, (int *)&v16);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v15);
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *(_DWORD *)v17 = v16;
  v3 = *(_DWORD *)(a2 + 16);
  v17 = (_QWORD *)(a2 + 16);
  LODWORD(v16) = v3;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "CBankOfstHi",
         0,
         0,
         &v14,
         &v15) )
  {
    sub_1C14060(a1, (int *)&v16);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v15);
  }
  result = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  if ( !(_BYTE)result )
  {
    result = (__int64)v17;
    *(_DWORD *)v17 = v16;
  }
  return result;
}
