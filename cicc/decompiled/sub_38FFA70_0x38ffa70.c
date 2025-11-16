// Function: sub_38FFA70
// Address: 0x38ffa70
//
__int64 __fastcall sub_38FFA70(
        __int64 a1,
        _BYTE *a2,
        __int64 a3,
        int a4,
        unsigned __int8 a5,
        unsigned int a6,
        __int64 a7,
        __int64 a8)
{
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v16; // rdi
  __int64 v17; // [rsp+8h] [rbp-68h]
  void (__fastcall *v18)(__int64, __int64, _QWORD); // [rsp+10h] [rbp-60h]
  const char *v20; // [rsp+20h] [rbp-50h] BYREF
  char v21; // [rsp+30h] [rbp-40h]
  char v22; // [rsp+31h] [rbp-3Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v17 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v18 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v17 + 160LL);
    v13 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v14 = sub_38C20E0(v13, a2, a3, a4, a5, a6, a7, a8, 0xFFFFFFFF, 0);
    v18(v17, v14, 0);
    return 0;
  }
  else
  {
    v16 = *(_QWORD *)(a1 + 8);
    v22 = 1;
    v20 = "unexpected token in section switching directive";
    v21 = 3;
    return sub_3909CF0(v16, &v20, 0, 0, v11, v12);
  }
}
