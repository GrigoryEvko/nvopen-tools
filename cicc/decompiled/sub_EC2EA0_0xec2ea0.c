// Function: sub_EC2EA0
// Address: 0xec2ea0
//
__int64 __fastcall sub_EC2EA0(
        __int64 a1,
        _BYTE *a2,
        size_t a3,
        unsigned int a4,
        __int64 a5,
        __int64 a6,
        unsigned __int8 a7)
{
  _QWORD *v10; // rax
  unsigned __int64 v11; // rax
  __int64 v13; // rdi
  __int64 v14; // [rsp+8h] [rbp-78h]
  void (__fastcall *v15)(__int64, unsigned __int64, _QWORD); // [rsp+10h] [rbp-70h]
  const char *v17; // [rsp+20h] [rbp-60h] BYREF
  char v18; // [rsp+40h] [rbp-40h]
  char v19; // [rsp+41h] [rbp-3Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    v14 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v15 = *(void (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)v14 + 176LL);
    v10 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v11 = sub_E6DEB0(v10, a2, a3, a4, a5, a6, a7, 0xFFFFFFFF);
    v15(v14, v11, 0);
    return 0;
  }
  else
  {
    v13 = *(_QWORD *)(a1 + 8);
    v19 = 1;
    v17 = "unexpected token in section switching directive";
    v18 = 3;
    return sub_ECE0E0(v13, &v17, 0, 0);
  }
}
