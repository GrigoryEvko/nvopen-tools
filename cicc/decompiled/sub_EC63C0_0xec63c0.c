// Function: sub_EC63C0
// Address: 0xec63c0
//
__int64 __fastcall sub_EC63C0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r13
  unsigned int v5; // r12d
  __int64 v6; // rax
  __int64 v8; // rdi
  __int64 v9; // rdi
  unsigned int v10; // [rsp+8h] [rbp-68h] BYREF
  const char *v11; // [rsp+10h] [rbp-60h] BYREF
  const char *v12; // [rsp+18h] [rbp-58h]
  const char *v13[4]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v14; // [rsp+40h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 8);
  v11 = 0;
  v12 = 0;
  if ( !(*(unsigned __int8 (__fastcall **)(__int64, const char **))(*(_QWORD *)v2 + 192LL))(v2, &v11) )
  {
    v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v14 = 261;
    v13[0] = v11;
    v13[1] = v12;
    v4 = sub_E6C460(v3, v13);
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 26 )
    {
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
      v5 = (*(__int64 (__fastcall **)(_QWORD, unsigned int *))(**(_QWORD **)(a1 + 8) + 256LL))(
             *(_QWORD *)(a1 + 8),
             &v10);
      if ( (_BYTE)v5 )
        return v5;
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
      {
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
        v6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
        (*(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v6 + 304LL))(v6, v4, v10);
        return v5;
      }
    }
    v8 = *(_QWORD *)(a1 + 8);
    v13[0] = "unexpected token in '.desc' directive";
    v14 = 259;
    return (unsigned int)sub_ECE0E0(v8, v13, 0, 0);
  }
  v9 = *(_QWORD *)(a1 + 8);
  v13[0] = "expected identifier in directive";
  v14 = 259;
  return (unsigned int)sub_ECE0E0(v9, v13, 0, 0);
}
