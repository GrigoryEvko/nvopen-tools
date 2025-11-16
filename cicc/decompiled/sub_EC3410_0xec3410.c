// Function: sub_EC3410
// Address: 0xec3410
//
__int64 __fastcall sub_EC3410(__int64 a1)
{
  __int64 v2; // rdi
  unsigned int v3; // eax
  unsigned int v4; // r12d
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rax
  const char *v9; // rax
  __int64 v10; // rdi
  const char *v11; // [rsp+0h] [rbp-60h] BYREF
  const char *v12; // [rsp+8h] [rbp-58h]
  const char *v13[4]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v14; // [rsp+30h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 8);
  v11 = 0;
  v12 = 0;
  v3 = (*(__int64 (__fastcall **)(__int64, const char **))(*(_QWORD *)v2 + 192LL))(v2, &v11);
  if ( (_BYTE)v3 )
  {
    HIBYTE(v14) = 1;
    v9 = "expected identifier in directive";
  }
  else
  {
    v4 = v3;
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
    {
      v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v14 = 261;
      v13[0] = v11;
      v13[1] = v12;
      v6 = sub_E6C460(v5, v13);
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
      v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v7 + 392LL))(v7, v6);
      return v4;
    }
    HIBYTE(v14) = 1;
    v9 = "unexpected token in directive";
  }
  v10 = *(_QWORD *)(a1 + 8);
  v13[0] = v9;
  LOBYTE(v14) = 3;
  return (unsigned int)sub_ECE0E0(v10, v13, 0, 0);
}
