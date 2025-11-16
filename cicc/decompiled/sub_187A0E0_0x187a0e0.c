// Function: sub_187A0E0
// Address: 0x187a0e0
//
__int64 __fastcall sub_187A0E0(__int64 a1, __int64 a2)
{
  unsigned __int8 (__fastcall *v3)(__int64, const char *, _BOOL8); // r15
  char v4; // al
  _BOOL8 v5; // rdx
  unsigned __int8 (__fastcall *v6)(__int64, const char *, _BOOL8); // r15
  char v7; // al
  _BOOL8 v8; // rdx
  unsigned __int8 (__fastcall *v9)(__int64, const char *, _BOOL8); // r15
  char v10; // al
  _BOOL8 v11; // rdx
  unsigned __int8 (__fastcall *v12)(__int64, const char *, _BOOL8); // r15
  char v13; // al
  _BOOL8 v14; // rdx
  char v15; // [rsp+7h] [rbp-39h] BYREF
  _QWORD v16[7]; // [rsp+8h] [rbp-38h] BYREF

  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Kind",
         0,
         0,
         &v15,
         v16) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v3 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v5 = 0;
    if ( v4 )
      v5 = *(_DWORD *)a2 == 0;
    if ( v3(a1, "Indir", v5) )
      *(_DWORD *)a2 = 0;
    v6 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v8 = 0;
    if ( v7 )
      v8 = *(_DWORD *)a2 == 1;
    if ( v6(a1, "UniformRetVal", v8) )
      *(_DWORD *)a2 = 1;
    v9 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v11 = 0;
    if ( v10 )
      v11 = *(_DWORD *)a2 == 2;
    if ( v9(a1, "UniqueRetVal", v11) )
      *(_DWORD *)a2 = 2;
    v12 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v13 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v14 = 0;
    if ( v13 )
      v14 = *(_DWORD *)a2 == 3;
    if ( v12(a1, "VirtualConstProp", v14) )
      *(_DWORD *)a2 = 3;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v16[0]);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Info",
         0,
         0,
         &v15,
         v16) )
  {
    sub_1879F60(a1, (_QWORD *)(a2 + 8));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v16[0]);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Byte",
         0,
         0,
         &v15,
         v16) )
  {
    sub_1879C60(a1, (unsigned int *)(a2 + 16));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v16[0]);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Bit",
         0,
         0,
         &v15,
         v16) )
  {
    sub_1879C60(a1, (unsigned int *)(a2 + 20));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v16[0]);
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
}
