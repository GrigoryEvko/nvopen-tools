// Function: sub_3902170
// Address: 0x3902170
//
__int64 __fastcall sub_3902170(
        __int64 a1,
        const void *a2,
        unsigned __int64 a3,
        const void *a4,
        unsigned __int64 a5,
        int a6,
        unsigned int a7,
        int a8)
{
  __int64 v12; // r8
  __int64 v13; // r9
  __int64 v14; // rax
  unsigned __int64 v15; // rdx
  unsigned int v16; // r12d
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v20; // rdi
  __int64 v21; // [rsp+8h] [rbp-68h]
  void (__fastcall *v22)(__int64, __int64, _QWORD); // [rsp+10h] [rbp-60h]
  const char *v23; // [rsp+20h] [rbp-50h] BYREF
  char v24; // [rsp+30h] [rbp-40h]
  char v25; // [rsp+31h] [rbp-3Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 136LL))(*(_QWORD *)(a1 + 8));
    v21 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v22 = *(void (__fastcall **)(__int64, __int64, _QWORD))(*(_QWORD *)v21 + 160LL);
    v14 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v15 = a3;
    v16 = 0;
    v17 = sub_38BFA90(v14, a2, v15, a4, a5, a6, a8, (unsigned __int8)(((a6 >> 31) & 0xF0) + 17), 0);
    v22(v21, v17, 0);
    if ( a7 )
    {
      v18 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      (*(void (__fastcall **)(__int64, _QWORD, _QWORD, __int64, _QWORD))(*(_QWORD *)v18 + 512LL))(v18, a7, 0, 1, 0);
    }
  }
  else
  {
    v20 = *(_QWORD *)(a1 + 8);
    v25 = 1;
    v23 = "unexpected token in section switching directive";
    v24 = 3;
    return (unsigned int)sub_3909CF0(v20, &v23, 0, 0, v12, v13);
  }
  return v16;
}
