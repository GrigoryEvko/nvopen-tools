// Function: sub_ECA590
// Address: 0xeca590
//
__int64 __fastcall sub_ECA590(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rax
  __int64 v4; // r13
  unsigned int v5; // r12d
  __int64 v6; // rax
  const char *v8; // rax
  __int64 v9; // rdi
  __int64 v10; // rdi
  __int64 v11; // [rsp+8h] [rbp-68h] BYREF
  const char *v12; // [rsp+10h] [rbp-60h] BYREF
  const char *v13; // [rsp+18h] [rbp-58h]
  const char *v14[4]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v15; // [rsp+40h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 8);
  v12 = 0;
  v13 = 0;
  if ( !(*(unsigned __int8 (__fastcall **)(__int64, const char **))(*(_QWORD *)v2 + 192LL))(v2, &v12) )
  {
    v3 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v15 = 261;
    v14[0] = v12;
    v14[1] = v13;
    v4 = sub_E6C460(v3, v14);
    if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 26 )
    {
      (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
      v5 = sub_ECD870(*(_QWORD *)(a1 + 8), &v11);
      if ( (_BYTE)v5 )
        return v5;
      if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
      {
        (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
        v6 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
        (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v6 + 448LL))(v6, v4, v11);
        return v5;
      }
      HIBYTE(v15) = 1;
      v8 = "unexpected token";
    }
    else
    {
      HIBYTE(v15) = 1;
      v8 = "expected comma";
    }
    v9 = *(_QWORD *)(a1 + 8);
    v14[0] = v8;
    LOBYTE(v15) = 3;
    return (unsigned int)sub_ECE0E0(v9, v14, 0, 0);
  }
  v10 = *(_QWORD *)(a1 + 8);
  v14[0] = "expected identifier";
  v15 = 259;
  return (unsigned int)sub_ECE0E0(v10, v14, 0, 0);
}
