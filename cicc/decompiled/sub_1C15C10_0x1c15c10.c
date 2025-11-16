// Function: sub_1C15C10
// Address: 0x1c15c10
//
__int64 __fastcall sub_1C15C10(__int64 a1, __int64 a2)
{
  char v2; // al
  _BOOL8 v3; // rcx
  char v4; // al
  _BOOL8 v5; // rcx
  char v6; // al
  _BOOL8 v7; // rcx
  char v8; // al
  unsigned __int8 v9; // al
  unsigned __int8 v10; // al
  char v11; // al
  char v12; // r10
  char v13; // al
  _BOOL8 v14; // rcx
  int v15; // eax
  __int64 result; // rax
  unsigned __int8 v17; // al
  bool v18; // zf
  __int64 v19; // rax
  char v20; // al
  _BOOL8 v21; // rdx
  char v22; // al
  __int64 v23; // rax
  char v24; // al
  _BOOL8 v25; // rdx
  char v26; // al
  __int64 v27; // rax
  char v28; // al
  _BOOL8 v29; // rdx
  char v30; // al
  char v31; // al
  unsigned __int8 v32; // [rsp+8h] [rbp-68h]
  unsigned __int8 (__fastcall *v33)(__int64, char *, _QWORD); // [rsp+8h] [rbp-68h]
  int v34; // [rsp+14h] [rbp-5Ch]
  bool v35; // [rsp+18h] [rbp-58h]
  __int64 (__fastcall *v36)(__int64, const char *, _BOOL8); // [rsp+18h] [rbp-58h]
  __int64 (__fastcall *v37)(__int64, char *, _BOOL8); // [rsp+18h] [rbp-58h]
  __int64 (__fastcall *v38)(__int64, const char *, _BOOL8); // [rsp+18h] [rbp-58h]
  char v39; // [rsp+18h] [rbp-58h]
  char v40; // [rsp+27h] [rbp-49h] BYREF
  __int64 v41; // [rsp+28h] [rbp-48h] BYREF
  __int64 v42; // [rsp+30h] [rbp-40h] BYREF
  _DWORD *v43; // [rsp+38h] [rbp-38h]

  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, __int64, _QWORD, __int64 *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Enabled",
         1,
         0,
         &v41,
         &v42) )
  {
    sub_1C14710(a1, (unsigned int *)a2);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v42);
  }
  LODWORD(v42) = *(_DWORD *)(a2 + 4);
  v43 = (_DWORD *)(a2 + 4);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "CbBankToReservedVABase",
         0,
         0,
         &v40,
         &v41) )
  {
    sub_1C14060(a1, (int *)&v42);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v41);
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *v43 = v42;
  LODWORD(v42) = *(_DWORD *)(a2 + 8);
  v43 = (_DWORD *)(a2 + 8);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "CbByteOffsetToReservedVABase",
         0,
         0,
         &v40,
         &v41) )
  {
    sub_1C14060(a1, (int *)&v42);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v41);
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *v43 = v42;
  LODWORD(v42) = *(_DWORD *)(a2 + 12);
  v43 = (_DWORD *)(a2 + 12);
  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v3 = 0;
  if ( v2 )
    v3 = (_DWORD)v42 == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "CbAddressBitsInReservedVABase",
         0,
         v3,
         &v40,
         &v41) )
  {
    sub_1C14060(a1, (int *)&v42);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v41);
  }
  else if ( v40 )
  {
    LODWORD(v42) = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *v43 = v42;
  LODWORD(v42) = *(_DWORD *)(a2 + 16);
  v43 = (_DWORD *)(a2 + 16);
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v5 = 0;
  if ( v4 )
    v5 = (_DWORD)v42 == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "CbBitShiftInReservedVABase",
         0,
         v5,
         &v40,
         &v41) )
  {
    sub_1C14060(a1, (int *)&v42);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v41);
  }
  else if ( v40 )
  {
    LODWORD(v42) = 0;
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *v43 = v42;
  LODWORD(v42) = *(_DWORD *)(a2 + 20);
  v43 = (_DWORD *)(a2 + 20);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ByteOffsetToStartOfReservedArea",
         0,
         0,
         &v40,
         &v41) )
  {
    sub_1C14060(a1, (int *)&v42);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v41);
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *v43 = v42;
  LODWORD(v42) = *(_DWORD *)(a2 + 24);
  v43 = (_DWORD *)(a2 + 24);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ByteOffsetToEndOfReservedArea",
         0,
         0,
         &v40,
         &v41) )
  {
    sub_1C14060(a1, (int *)&v42);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v41);
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *v43 = v42;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, __int64, _QWORD, __int64 *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ReservedCbReadBank",
         1,
         0,
         &v41,
         &v42) )
  {
    sub_1C14710(a1, (unsigned int *)(a2 + 28));
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v42);
  }
  LODWORD(v42) = *(_DWORD *)(a2 + 32);
  v43 = (_DWORD *)(a2 + 32);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ReservedCbReadByteOffset",
         0,
         0,
         &v40,
         &v41) )
  {
    sub_1C14060(a1, (int *)&v42);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v41);
  }
  if ( !(*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) )
    *v43 = v42;
  LODWORD(v41) = *(_BYTE *)(a2 + 36) & 1;
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v7 = 0;
  if ( v6 )
    v7 = (_DWORD)v41 == 0;
  v8 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ForceHighLatencyConstExpr",
         0,
         v7,
         &v40,
         &v42);
  if ( v8 )
  {
    sub_1C14710(a1, (unsigned int *)&v41);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v42);
    v8 = v41 & 1;
  }
  else if ( !v40 )
  {
    v8 = v41 & 1;
  }
  v9 = *(_BYTE *)(a2 + 36) & 0xFE | v8 & 1;
  *(_BYTE *)(a2 + 36) = v9;
  v32 = (v9 >> 1) & 7;
  v34 = v32;
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v35 = v32 == 0;
  v11 = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __int64 *, __int64 *))(*(_QWORD *)a1 + 120LL))(
          a1,
          "SetupSubRoutineRetKind",
          0,
          v35 & v10,
          &v41,
          &v42);
  v12 = v32;
  if ( v11 )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v33 = *(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD))(*(_QWORD *)a1 + 168LL);
    v17 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v18 = v33(a1, "Default", v35 & v17) == 0;
    v19 = *(_QWORD *)a1;
    v36 = *(__int64 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    if ( v18 )
    {
      v20 = (*(__int64 (__fastcall **)(__int64))(v19 + 16))(a1);
      v21 = 0;
      if ( v20 )
        v21 = v34 == 1;
      v22 = v36(a1, "NoDec", v21);
    }
    else
    {
      (*(void (__fastcall **)(__int64))(v19 + 16))(a1);
      v34 = 0;
      v22 = v36(a1, "NoDec", 0);
    }
    v18 = v22 == 0;
    v23 = *(_QWORD *)a1;
    v37 = *(__int64 (__fastcall **)(__int64, char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    if ( v18 )
    {
      v24 = (*(__int64 (__fastcall **)(__int64))(v23 + 16))(a1);
      v25 = 0;
      if ( v24 )
        v25 = v34 == 2;
      v26 = v37(a1, "Exit", v25);
    }
    else
    {
      (*(void (__fastcall **)(__int64))(v23 + 16))(a1);
      v34 = 1;
      v26 = v37(a1, "Exit", 0);
    }
    v18 = v26 == 0;
    v27 = *(_QWORD *)a1;
    v38 = *(__int64 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    if ( v18 )
    {
      v28 = (*(__int64 (__fastcall **)(__int64))(v27 + 16))(a1);
      v29 = 0;
      if ( v28 )
        v29 = v34 == 3;
      v30 = v38(a1, "Invalid", v29);
    }
    else
    {
      (*(void (__fastcall **)(__int64))(v27 + 16))(a1);
      LOBYTE(v34) = 2;
      v30 = v38(a1, "Invalid", 0);
    }
    v18 = v30 == 0;
    v31 = 3;
    if ( v18 )
      v31 = v34;
    v39 = v31;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v42);
    v12 = v39;
  }
  else if ( (_BYTE)v41 )
  {
    v12 = 0;
  }
  *(_BYTE *)(a2 + 36) = *(_BYTE *)(a2 + 36) & 0xF1 | (2 * (v12 & 7));
  LODWORD(v41) = *(_DWORD *)(a2 + 36) >> 4;
  v13 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v14 = 0;
  if ( v13 )
    v14 = (_DWORD)v41 == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _BOOL8, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Reserved",
         0,
         v14,
         &v40,
         &v42) )
  {
    sub_1C14710(a1, (unsigned int *)&v41);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v42);
    v15 = v41 & 0xFFFFFFF;
  }
  else
  {
    v15 = 0;
    if ( !v40 )
      v15 = v41 & 0xFFFFFFF;
  }
  result = (16 * v15) | *(_DWORD *)(a2 + 36) & 0xFu;
  *(_DWORD *)(a2 + 36) = result;
  return result;
}
