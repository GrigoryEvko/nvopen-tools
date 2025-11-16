// Function: sub_262BF20
// Address: 0x262bf20
//
__int64 __fastcall sub_262BF20(__int64 a1, __int64 a2)
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
  _BYTE *v15; // rsi
  unsigned __int64 v16; // rbx
  unsigned __int64 v17; // r14
  unsigned __int64 v18; // rdx
  char v19; // [rsp+7h] [rbp-59h] BYREF
  __int64 v20; // [rsp+8h] [rbp-58h] BYREF
  unsigned __int64 v21; // [rsp+10h] [rbp-50h] BYREF
  unsigned __int64 v22; // [rsp+18h] [rbp-48h]
  __int64 v23; // [rsp+20h] [rbp-40h]

  (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 104LL))(a1);
  if ( (*(unsigned __int8 (__fastcall **)(__int64, char *, _QWORD, _QWORD, __int64 *, unsigned __int64 *))(*(_QWORD *)a1 + 120LL))(
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
      v5 = *(_DWORD *)a2 == 0;
    if ( v3(a1, "Indir", v5) )
      *(_DWORD *)a2 = 0;
    v6 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v7 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v8 = 0;
    if ( v7 )
      v8 = *(_DWORD *)a2 == 1;
    if ( v6(a1, "SingleImpl", v8) )
      *(_DWORD *)a2 = 1;
    v9 = *(unsigned __int8 (__fastcall **)(__int64, const char *, _BOOL8))(*(_QWORD *)a1 + 168LL);
    v10 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1);
    v11 = 0;
    if ( v10 )
      v11 = *(_DWORD *)a2 == 2;
    if ( v9(a1, "BranchFunnel", v11) )
      *(_DWORD *)a2 = 2;
    (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 184LL))(a1);
    (*(void (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)a1 + 128LL))(a1, v21);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, __int64 *, unsigned __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "SingleImplName",
         0,
         0,
         &v20,
         &v21) )
  {
    sub_2625F10(a1, (_QWORD *)(a2 + 8));
    (*(void (__fastcall **)(__int64, unsigned __int64))(*(_QWORD *)a1 + 128LL))(a1, v21);
  }
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char *, _QWORD, _QWORD, char *, __int64 *))(*(_QWORD *)a1 + 120LL))(
         a1,
         "ResByArg",
         0,
         0,
         &v19,
         &v20) )
  {
    v12 = (_QWORD *)(a2 + 40);
    v13 = (*(unsigned __int8 (__fastcall **)(__int64))(*(_QWORD *)a1 + 16LL))(a1) == 0;
    v14 = *(_QWORD *)a1;
    if ( v13 )
    {
      (*(void (__fastcall **)(__int64))(v14 + 104))(a1);
      v15 = (_BYTE *)a1;
      (*(void (__fastcall **)(unsigned __int64 *, __int64))(*(_QWORD *)a1 + 136LL))(&v21, a1);
      v16 = v21;
      v17 = v22;
      if ( v21 != v22 )
      {
        do
        {
          v15 = *(_BYTE **)v16;
          v18 = *(_QWORD *)(v16 + 8);
          v16 += 16LL;
          sub_262B910(a1, v15, v18, v12);
        }
        while ( v17 != v16 );
        v17 = v21;
      }
      if ( v17 )
      {
        v15 = (_BYTE *)(v23 - v17);
        j_j___libc_free_0(v17);
      }
      (*(void (__fastcall **)(__int64, _BYTE *))(*(_QWORD *)a1 + 112LL))(a1, v15);
    }
    else
    {
      (*(void (__fastcall **)(__int64))(v14 + 104))(a1);
      sub_2626470(a1, (__int64)v12);
      (*(void (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
    }
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)a1 + 128LL))(a1, v20);
  }
  return (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)a1 + 112LL))(a1);
}
