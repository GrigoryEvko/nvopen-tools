// Function: sub_1882410
// Address: 0x1882410
//
__int64 __fastcall sub_1882410(__int64 a1, _DWORD *a2)
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
  _QWORD *v12; // r13
  bool v13; // zf
  __int64 v14; // rax
  __int64 v15; // rbx
  __int64 v16; // r14
  _BYTE *v17; // rsi
  unsigned __int64 v18; // rdx
  char v19; // [rsp+7h] [rbp-59h] BYREF
  __int64 v20; // [rsp+8h] [rbp-58h] BYREF
  __int64 v21; // [rsp+10h] [rbp-50h] BYREF
  __int64 v22; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+20h] [rbp-40h]

  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, __int64 *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "Kind",
         0,
         0,
         &v20,
         &v21) )
  {
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 160LL))(a1);
    v3 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v4 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v5 = 0;
    if ( v4 )
      v5 = *a2 == 0;
    if ( v3(a1, "Indir", v5) )
      *a2 = 0;
    v6 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v8 = 0;
    if ( v7 )
      v8 = *a2 == 1;
    if ( v6(a1, "SingleImpl", v8) )
      *a2 = 1;
    v9 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v11 = 0;
    if ( v10 )
      v11 = *a2 == 2;
    if ( v9(a1, "BranchFunnel", v11) )
      *a2 = 2;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v21);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __int64 *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "SingleImplName",
         0,
         0,
         &v20,
         &v21) )
  {
    sub_187CB10(a1, (__int64)(a2 + 2));
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v21);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ResByArg",
         0,
         0,
         &v19,
         &v20) )
  {
    v12 = a2 + 10;
    v13 = (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) == 0;
    v14 = *(_QWORD *)a1;
    if ( v13 )
    {
      (*(void (__fastcall **)(__int64))(v14 + 104))(a1);
      (*(void (__fastcall **)(__int64 *, __int64))(*(_QWORD *)a1 + 136LL))(&v21, a1);
      v15 = v21;
      v16 = v22;
      if ( v21 != v22 )
      {
        do
        {
          v17 = *(_BYTE **)v15;
          v18 = *(_QWORD *)(v15 + 8);
          v15 += 16;
          sub_1881E40(a1, v17, v18, v12);
        }
        while ( v16 != v15 );
        v16 = v21;
      }
      if ( v16 )
        j_j___libc_free_0(v16, v23 - v16);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
    }
    else
    {
      (*(void (__fastcall **)(__int64))(v14 + 104))(a1);
      sub_187D100(a1, (__int64)v12);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
    }
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v20);
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
}
