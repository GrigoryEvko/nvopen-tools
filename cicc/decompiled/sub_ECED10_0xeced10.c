// Function: sub_ECED10
// Address: 0xeced10
//
__int64 __fastcall sub_ECED10(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  __int64 v7; // rax
  __int64 v8; // r15
  unsigned int v9; // r12d
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rdi
  __int64 v14; // [rsp+8h] [rbp-78h] BYREF
  const char *v15; // [rsp+10h] [rbp-70h] BYREF
  const char *v16; // [rsp+18h] [rbp-68h]
  const char *v17[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v18; // [rsp+40h] [rbp-40h]

  v6 = *(_QWORD *)(a1 + 24);
  v15 = 0;
  v16 = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char **))(*(_QWORD *)v6 + 192LL))(v6, &v15) )
  {
    v13 = *(_QWORD *)(a1 + 8);
    v17[0] = "expected identifier in directive";
    v18 = 259;
    return (unsigned int)sub_ECE0E0(v13, (__int64)v17, 0, 0);
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v18 = 261;
  v17[0] = v15;
  v17[1] = v16;
  v8 = sub_E6C460(v7, v17);
  if ( **(_DWORD **)(*(_QWORD *)(a1 + 32) + 8LL) == 26 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  }
  else if ( (unsigned __int8)sub_ECEAE0(a1, ",") )
  {
    return 1;
  }
  v9 = sub_ECD870(*(__int64 **)(a1 + 24), (__int64)&v14);
  if ( (_BYTE)v9 )
    return 1;
  if ( **(_DWORD **)(*(_QWORD *)(a1 + 32) + 8LL) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
  }
  else if ( (unsigned __int8)sub_ECEAE0(a1, "eol") )
  {
    return 1;
  }
  v11 = *(_QWORD *)(a1 + 8);
  if ( *(_BYTE *)(v8 + 36) && !*(_DWORD *)(v8 + 32) )
  {
    v17[0] = ".size directive ignored for function symbols";
    v18 = 259;
    (*(void (__fastcall **)(__int64, __int64, const char **, _QWORD, _QWORD))(*(_QWORD *)v11 + 168LL))(
      v11,
      a4,
      v17,
      0,
      0);
  }
  else
  {
    v12 = (*(__int64 (__fastcall **)(__int64))(*(_QWORD *)v11 + 56LL))(v11);
    (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v12 + 448LL))(v12, v8, v14);
  }
  return v9;
}
