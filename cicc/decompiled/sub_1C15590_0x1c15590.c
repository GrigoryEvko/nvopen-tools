// Function: sub_1C15590
// Address: 0x1c15590
//
__int64 __fastcall sub_1C15590(__int64 a1, __int64 a2)
{
  char v2; // al
  _BOOL8 v3; // rcx
  char v4; // al
  __int64 v5; // rcx
  char v6; // al
  __int64 v7; // rcx
  char v8; // al
  _BOOL8 v9; // rcx
  char v10; // al
  _BOOL8 v11; // rcx
  char v12; // al
  _BOOL8 v13; // rcx
  __int64 result; // rax
  unsigned __int8 (__fastcall *v15)(__int64, const char *, _BOOL8); // r15
  char v16; // al
  _BOOL8 v17; // rdx
  unsigned __int8 (__fastcall *v18)(__int64, const char *, _BOOL8); // r15
  char v19; // al
  _BOOL8 v20; // rdx
  unsigned __int8 (__fastcall *v21)(__int64, const char *, _BOOL8); // r15
  char v22; // al
  _BOOL8 v23; // rdx
  char v24; // [rsp+7h] [rbp-39h] BYREF
  _QWORD v25[7]; // [rsp+8h] [rbp-38h] BYREF

  v2 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v3 = 0;
  if ( v2 )
    v3 = *(_BYTE *)a2 == 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "FastGsCodeGenType",
         0,
         v3,
         &v24,
         v25) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v15 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v16 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v17 = 0;
    if ( v16 )
      v17 = *(_BYTE *)a2 == 0;
    if ( v15(a1, "NoFastGs", v17) )
      *(_BYTE *)a2 = 0;
    v18 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v19 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v20 = 0;
    if ( v19 )
      v20 = *(_BYTE *)a2 == 1;
    if ( v18(a1, "ImplicitFastGs", v20) )
      *(_BYTE *)a2 = 1;
    v21 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v22 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v23 = 0;
    if ( v22 )
      v23 = *(_BYTE *)a2 == 2;
    if ( v21(a1, "ExplicitFastGs", v23) )
      *(_BYTE *)a2 = 2;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v25[0]);
  }
  else if ( v24 )
  {
    *(_BYTE *)a2 = 0;
  }
  v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v5 = 0;
  if ( v4 )
    v5 = *(_BYTE *)(a2 + 1) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, __int64, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "UseViewportMask",
         0,
         v5,
         &v24,
         v25) )
  {
    sub_1C14360(a1, (_BYTE *)(a2 + 1));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v25[0]);
  }
  else if ( v24 )
  {
    *(_BYTE *)(a2 + 1) = 0;
  }
  v6 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v7 = 0;
  if ( v6 )
    v7 = *(_BYTE *)(a2 + 2) ^ 1u;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, __int64, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "VREnabled",
         0,
         v7,
         &v24,
         v25) )
  {
    sub_1C14360(a1, (_BYTE *)(a2 + 2));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v25[0]);
  }
  else if ( v24 )
  {
    *(_BYTE *)(a2 + 2) = 0;
  }
  v8 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v9 = 0;
  if ( v8 )
    v9 = *(_DWORD *)(a2 + 4) == -1;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "VertexCount",
         0,
         v9,
         &v24,
         v25) )
  {
    sub_1C14590(a1, (int *)(a2 + 4));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v25[0]);
  }
  else if ( v24 )
  {
    *(_DWORD *)(a2 + 4) = -1;
  }
  v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v11 = 0;
  if ( v10 )
    v11 = *(_DWORD *)(a2 + 8) == -1;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ProvokingVertex",
         0,
         v11,
         &v24,
         v25) )
  {
    sub_1C14590(a1, (int *)(a2 + 8));
    (*(void (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v25[0]);
  }
  else if ( v24 )
  {
    *(_DWORD *)(a2 + 8) = -1;
  }
  v12 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
  v13 = 0;
  if ( v12 )
    v13 = *(_DWORD *)(a2 + 12) == -1;
  result = (*(__int64 (__fastcall **)(__int64, const char *, _QWORD, _BOOL8, char *, _QWORD *))(*(_QWORD *)a1 + 120LL))(
             a1,
             "InstanceCount",
             0,
             v13,
             &v24,
             v25);
  if ( (_BYTE)result )
  {
    sub_1C14590(a1, (int *)(a2 + 12));
    return (*(__int64 (__fastcall **)(__int64, _QWORD))(*(_QWORD *)a1 + 128LL))(a1, v25[0]);
  }
  else if ( v24 )
  {
    *(_DWORD *)(a2 + 12) = -1;
  }
  return result;
}
