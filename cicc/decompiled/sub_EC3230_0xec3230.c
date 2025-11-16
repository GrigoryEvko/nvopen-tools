// Function: sub_EC3230
// Address: 0xec3230
//
__int64 __fastcall sub_EC3230(__int64 a1)
{
  __int64 v2; // rdi
  unsigned int v3; // eax
  unsigned int v4; // r12d
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v9; // rdi
  const char *v10; // [rsp+0h] [rbp-60h] BYREF
  const char *v11; // [rsp+8h] [rbp-58h]
  const char *v12[4]; // [rsp+10h] [rbp-50h] BYREF
  __int16 v13; // [rsp+30h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 8);
  v10 = 0;
  v11 = 0;
  v3 = (*(__int64 (__fastcall **)(__int64, const char **))(*(_QWORD *)v2 + 192LL))(v2, &v10);
  if ( (_BYTE)v3 )
  {
    v9 = *(_QWORD *)(a1 + 8);
    v12[0] = "expected identifier in directive";
    v13 = 259;
    return (unsigned int)sub_ECE0E0(v9, v12, 0, 0);
  }
  else
  {
    v4 = v3;
    v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v13 = 261;
    v12[0] = v10;
    v12[1] = v11;
    v6 = sub_E6C460(v5, v12);
    v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v7 + 312LL))(v7, v6);
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    return v4;
  }
}
