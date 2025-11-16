// Function: sub_ECA710
// Address: 0xeca710
//
__int64 __fastcall sub_ECA710(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // rdi
  unsigned int v4; // r12d
  __int64 v5; // rax
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // r14
  __int64 v9; // rax
  const char *v11; // rax
  __int64 v12; // rdi
  const char *v13; // [rsp+0h] [rbp-70h] BYREF
  __int64 v14; // [rsp+8h] [rbp-68h]
  const char *v15; // [rsp+10h] [rbp-60h] BYREF
  __int64 v16; // [rsp+18h] [rbp-58h]
  const char *v17; // [rsp+20h] [rbp-50h] BYREF
  __int64 v18; // [rsp+28h] [rbp-48h]
  __int16 v19; // [rsp+40h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 8);
  v13 = 0;
  v14 = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, const char **))(*(_QWORD *)v2 + 192LL))(v2, &v13) )
  {
LABEL_7:
    HIBYTE(v19) = 1;
    v11 = "expected identifier";
    goto LABEL_6;
  }
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 26 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    v3 = *(_QWORD *)(a1 + 8);
    v15 = 0;
    v16 = 0;
    v4 = (*(__int64 (__fastcall **)(__int64, const char **))(*(_QWORD *)v3 + 192LL))(v3, &v15);
    if ( !(_BYTE)v4 )
    {
      v5 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v19 = 261;
      v17 = v13;
      v18 = v14;
      v6 = sub_E6C460(v5, &v17);
      v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
      v19 = 261;
      v17 = v15;
      v18 = v16;
      v8 = sub_E6C460(v7, &v17);
      v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v9 + 288LL))(v9, v6, v8);
      return v4;
    }
    goto LABEL_7;
  }
  HIBYTE(v19) = 1;
  v11 = "expected a comma";
LABEL_6:
  v12 = *(_QWORD *)(a1 + 8);
  v17 = v11;
  LOBYTE(v19) = 3;
  return (unsigned int)sub_ECE0E0(v12, &v17, 0, 0);
}
