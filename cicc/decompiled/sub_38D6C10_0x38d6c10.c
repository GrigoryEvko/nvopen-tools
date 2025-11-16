// Function: sub_38D6C10
// Address: 0x38d6c10
//
char __fastcall sub_38D6C10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, unsigned __int8 a5)
{
  bool (__fastcall *v6)(__int64, __int64, __int64, __int64); // r10
  unsigned __int64 v7; // r12
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  unsigned __int64 v12; // rax
  unsigned __int8 v13; // [rsp+4h] [rbp-3Ch]
  bool (__fastcall *v14)(__int64, __int64, __int64, __int64); // [rsp+8h] [rbp-38h]

  v6 = *(bool (__fastcall **)(__int64, __int64, __int64, __int64))(*(_QWORD *)a1 + 48LL);
  v7 = *(_QWORD *)a4 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v7 || (v7 = 0, (*(_BYTE *)(a4 + 9) & 0xC) != 8) )
  {
    if ( v6 == sub_38D6BB0 )
      goto LABEL_4;
    return ((__int64 (__fastcall *)(__int64, __int64, __int64, unsigned __int64, _QWORD, _QWORD))v6)(
             a1,
             a2,
             a3,
             v7,
             a5,
             0);
  }
  *(_BYTE *)(a4 + 8) |= 4u;
  v13 = a5;
  v14 = v6;
  v12 = (unsigned __int64)sub_38CE440(*(_QWORD *)(a4 + 24));
  v6 = v14;
  a5 = v13;
  v7 = v12;
  *(_QWORD *)a4 = v12 | *(_QWORD *)a4 & 7LL;
  if ( v14 != sub_38D6BB0 )
    return ((__int64 (__fastcall *)(__int64, __int64, __int64, unsigned __int64, _QWORD, _QWORD))v6)(
             a1,
             a2,
             a3,
             v7,
             a5,
             0);
LABEL_4:
  v9 = *(_QWORD *)a3 & 0xFFFFFFFFFFFFFFF8LL;
  if ( v9
    || (*(_BYTE *)(a3 + 9) & 0xC) == 8
    && (*(_BYTE *)(a3 + 8) |= 4u,
        v9 = (unsigned __int64)sub_38CE440(*(_QWORD *)(a3 + 24)),
        *(_QWORD *)a3 = v9 | *(_QWORD *)a3 & 7LL,
        v9) )
  {
    v10 = *(_QWORD *)(v9 + 24);
  }
  else
  {
    v10 = 0;
  }
  return *(_QWORD *)(v7 + 24) == v10;
}
