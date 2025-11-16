// Function: sub_2626120
// Address: 0x2626120
//
__int64 __fastcall sub_2626120(__int64 a1, __int64 a2)
{
  __int64 result; // rax
  unsigned __int8 (__fastcall *v3)(__int64, const char *, _BOOL8); // r15
  char v4; // al
  _BOOL8 v5; // rdx
  unsigned __int8 (__fastcall *v6)(__int64, const char *, _BOOL8); // r15
  char v7; // al
  _BOOL8 v8; // rdx
  unsigned __int8 (__fastcall *v9)(__int64, const char *, _BOOL8); // r15
  char v10; // al
  _BOOL8 v11; // rdx
  unsigned __int8 (__fastcall *v12)(__int64, char *, _BOOL8); // r15
  char v13; // al
  _BOOL8 v14; // rdx
  unsigned __int8 (__fastcall *v15)(__int64, const char *, _BOOL8); // r15
  char v16; // al
  _BOOL8 v17; // rdx
  unsigned __int8 (__fastcall *v18)(__int64, const char *, _BOOL8); // r15
  char v19; // al
  _BOOL8 v20; // rdx
  char v21; // [rsp+7h] [rbp-39h] BYREF
  _QWORD v22[7]; // [rsp+8h] [rbp-38h] BYREF

  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Kind",
         0,
         0,
         &v21,
         v22) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v3 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v5 = 0;
    if ( v4 )
      v5 = *(_DWORD *)a2 == 5;
    if ( v3(a1, "Unknown", v5) )
      *(_DWORD *)a2 = 5;
    v6 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v8 = 0;
    if ( v7 )
      v8 = *(_DWORD *)a2 == 0;
    if ( v6(a1, "Unsat", v8) )
      *(_DWORD *)a2 = 0;
    v9 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v11 = 0;
    if ( v10 )
      v11 = *(_DWORD *)a2 == 1;
    if ( v9(a1, "ByteArray", v11) )
      *(_DWORD *)a2 = 1;
    v12 = *(unsigned __int8 (__fastcall **)(__int64, char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v13 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v14 = 0;
    if ( v13 )
      v14 = *(_DWORD *)a2 == 2;
    if ( v12(a1, "Inline", v14) )
      *(_DWORD *)a2 = 2;
    v15 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v16 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v17 = 0;
    if ( v16 )
      v17 = *(_DWORD *)a2 == 3;
    if ( v15(a1, "Single", v17) )
      *(_DWORD *)a2 = 3;
    v18 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v19 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v20 = 0;
    if ( v19 )
      v20 = *(_DWORD *)a2 == 4;
    if ( v18(a1, "AllOnes", v20) )
      *(_DWORD *)a2 = 4;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v22[0]);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "SizeM1BitWidth",
         0,
         0,
         &v21,
         v22) )
  {
    sub_261B850(a1, (unsigned int *)(a2 + 4));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v22[0]);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "AlignLog2",
         0,
         0,
         &v21,
         v22) )
  {
    sub_261BC10(a1, (_QWORD *)(a2 + 8));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v22[0]);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "SizeM1",
         0,
         0,
         &v21,
         v22) )
  {
    sub_261BC10(a1, (_QWORD *)(a2 + 16));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v22[0]);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "BitMask",
         0,
         0,
         &v21,
         v22) )
  {
    sub_261BA30(a1, (unsigned __int8 *)(a2 + 24));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v22[0]);
  }
  result = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "InlineBits",
             0,
             0,
             &v21,
             v22);
  if ( (_BYTE)result )
  {
    sub_261BC10(a1, (_QWORD *)(a2 + 32));
    return (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v22[0]);
  }
  return result;
}
