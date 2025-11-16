// Function: sub_EC4350
// Address: 0xec4350
//
__int64 __fastcall sub_EC4350(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rdi
  unsigned int v7; // r12d
  const char *v8; // rax
  __int64 v9; // rdi
  __int64 v11; // rax
  __int64 v12; // r14
  __int64 v13; // rax
  unsigned __int8 v14; // [rsp+Eh] [rbp-72h] BYREF
  unsigned __int8 v15; // [rsp+Fh] [rbp-71h] BYREF
  const char *v16; // [rsp+10h] [rbp-70h] BYREF
  const char *v17; // [rsp+18h] [rbp-68h]
  const char *v18[4]; // [rsp+20h] [rbp-60h] BYREF
  __int16 v19; // [rsp+40h] [rbp-40h]

  v6 = *(_QWORD *)(a1 + 8);
  v16 = 0;
  v17 = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char **))(*(_QWORD *)v6 + 192LL))(v6, &v16) )
    return 1;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 26 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    v14 = 0;
    v15 = 0;
    v7 = sub_EC3910(a1, &v14, &v15);
    if ( (_BYTE)v7 )
      return 1;
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 26 )
    {
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
      if ( (unsigned __int8)sub_EC3910(a1, &v14, &v15) )
        return 1;
    }
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
    {
      v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v19 = 261;
      v18[0] = v16;
      v18[1] = v17;
      v12 = sub_E6C460(v11, v18);
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
      v13 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      (*(void (__fastcall **)(__int64, __int64, _QWORD, _QWORD, __int64))(*(_QWORD *)v13 + 1168LL))(
        v13,
        v12,
        v14,
        v15,
        a4);
      return v7;
    }
    HIBYTE(v19) = 1;
    v8 = "unexpected token in directive";
  }
  else
  {
    HIBYTE(v19) = 1;
    v8 = "you must specify one or both of @unwind or @except";
  }
  v9 = *(_QWORD *)(a1 + 8);
  v18[0] = v8;
  LOBYTE(v19) = 3;
  return (unsigned int)sub_ECE0E0(v9, v18, 0, 0);
}
