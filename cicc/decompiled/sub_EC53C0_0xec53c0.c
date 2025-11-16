// Function: sub_EC53C0
// Address: 0xec53c0
//
__int64 __fastcall sub_EC53C0(__int64 a1, __int64 a2, __int64 a3, void *a4, void *a5, int a6, unsigned int a7, int a8)
{
  __int64 v12; // rax
  __int64 v13; // rdx
  unsigned int v14; // r12d
  unsigned __int64 v15; // rax
  _QWORD *v16; // rax
  __int64 v17; // r9
  _QWORD *v18; // rdi
  __int64 v20; // rdi
  __int64 v21; // [rsp+8h] [rbp-78h]
  void (__fastcall *v22)(__int64, unsigned __int64, _QWORD); // [rsp+10h] [rbp-70h]
  const char *v23; // [rsp+20h] [rbp-60h] BYREF
  char v24; // [rsp+40h] [rbp-40h]
  char v25; // [rsp+41h] [rbp-3Fh]

  if ( **(_DWORD **)((*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 40LL))(*(_QWORD *)(a1 + 8)) + 8) == 9 )
  {
    (*(void (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 184LL))(*(_QWORD *)(a1 + 8));
    v21 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
    v22 = *(void (__fastcall **)(__int64, unsigned __int64, _QWORD))(*(_QWORD *)v21 + 176LL);
    v12 = (*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 48LL))(*(_QWORD *)(a1 + 8));
    v13 = a3;
    v14 = 0;
    v15 = sub_E6D970(v12, a2, v13, a4, a5, a6, a8, ((a6 >> 31) & 0xEFu) + 19, 0);
    v22(v21, v15, 0);
    if ( a7 )
    {
      v16 = (_QWORD *)(*(__int64 (__fastcall **)(_QWORD))(**(_QWORD **)(a1 + 8) + 56LL))(*(_QWORD *)(a1 + 8));
      v17 = *v16;
      v18 = v16;
      _BitScanReverse64((unsigned __int64 *)&v16, a7);
      (*(void (__fastcall **)(_QWORD *, _QWORD, _QWORD, __int64, _QWORD))(v17 + 608))(
        v18,
        63 - ((unsigned int)v16 ^ 0x3F),
        0,
        1,
        0);
    }
  }
  else
  {
    v20 = *(_QWORD *)(a1 + 8);
    v25 = 1;
    v23 = "unexpected token in section switching directive";
    v24 = 3;
    return (unsigned int)sub_ECE0E0(v20, &v23, 0, 0);
  }
  return v14;
}
