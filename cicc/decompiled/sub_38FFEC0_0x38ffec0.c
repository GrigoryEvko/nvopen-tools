// Function: sub_38FFEC0
// Address: 0x38ffec0
//
__int64 __fastcall sub_38FFEC0(__int64 a1)
{
  __int64 v2; // rdi
  unsigned int v3; // eax
  __int64 v4; // r8
  __int64 v5; // r9
  unsigned int v6; // r12d
  __int64 v7; // rax
  __int64 v8; // r13
  __int64 v9; // rax
  __int64 v11; // rdi
  _QWORD v12[2]; // [rsp+0h] [rbp-50h] BYREF
  _QWORD v13[2]; // [rsp+10h] [rbp-40h] BYREF
  __int16 v14; // [rsp+20h] [rbp-30h]

  v2 = *(_QWORD *)(a1 + 8);
  v12[0] = 0;
  v12[1] = 0;
  v3 = (*(__int64 (__fastcall **)(__int64, _QWORD *))(*(_QWORD *)v2 + 144LL))(v2, v12);
  if ( (_BYTE)v3 )
  {
    v11 = *(_QWORD *)(a1 + 8);
    v13[0] = "expected identifier in directive";
    v14 = 259;
    return (unsigned int)sub_3909CF0(v11, v13, 0, 0, v4, v5);
  }
  else
  {
    v6 = v3;
    v7 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v13[0] = v12;
    v14 = 261;
    v8 = sub_38BF510(v7, (__int64)v13);
    v9 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    (*(void (__fastcall **)(__int64, __int64))(*(_QWORD *)v9 + 272LL))(v9, v8);
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    return v6;
  }
}
