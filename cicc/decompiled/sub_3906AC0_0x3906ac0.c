// Function: sub_3906AC0
// Address: 0x3906ac0
//
__int64 __fastcall sub_3906AC0(__int64 a1)
{
  __int64 v2; // rdi
  __int64 v3; // r8
  __int64 v4; // r9
  __int64 v5; // rdi
  unsigned int v6; // r12d
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v10; // r14
  __int64 v11; // rax
  const char *v13; // rax
  __int64 v14; // rdi
  _QWORD v15[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v16[2]; // [rsp+10h] [rbp-60h] BYREF
  _QWORD v17[2]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v18; // [rsp+30h] [rbp-40h]

  v2 = *(_QWORD *)(a1 + 8);
  v15[0] = 0;
  v15[1] = 0;
  if ( (*(unsigned __int8 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v2 + 144LL))(v2, v15) )
    goto LABEL_8;
  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) != 25 )
  {
    HIBYTE(v18) = 1;
    v13 = "expected a comma";
LABEL_7:
    v14 = *(_QWORD *)(a1 + 8);
    v17[0] = v13;
    LOBYTE(v18) = 3;
    return (unsigned int)sub_3909CF0(v14, v17, 0, 0, v3, v4);
  }
  (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
  v5 = *(_QWORD *)(a1 + 8);
  v16[0] = 0;
  v16[1] = 0;
  v6 = (*(__int64 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v5 + 144LL))(v5, v16);
  if ( (_BYTE)v6 )
  {
LABEL_8:
    HIBYTE(v18) = 1;
    v13 = "expected identifier in directive";
    goto LABEL_7;
  }
  v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v17[0] = v15;
  v18 = 261;
  v8 = sub_38BF510(v7, (__int64)v17);
  v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
  v17[0] = v16;
  v18 = 261;
  v10 = sub_38BF510(v9, (__int64)v17);
  v11 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
  (*(void (__fastcall **)(__int64, __int64, __int64))(*(_QWORD *)v11 + 248LL))(v11, v8, v10);
  return v6;
}
