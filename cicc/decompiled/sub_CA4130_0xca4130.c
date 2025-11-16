// Function: sub_CA4130
// Address: 0xca4130
//
__int64 __fastcall sub_CA4130(
        __int64 a1,
        __int64 *a2,
        __int64 a3,
        __int64 a4,
        unsigned __int8 a5,
        unsigned __int8 a6,
        char a7)
{
  __int64 v11; // rax
  int v13; // eax
  _QWORD v14[2]; // [rsp+0h] [rbp-50h] BYREF
  char v15; // [rsp+10h] [rbp-40h]

  v11 = *a2;
  if ( !a7 )
  {
    (*(void (__fastcall **)(_QWORD *))(v11 + 56))(v14);
    if ( (v15 & 1) == 0 )
      goto LABEL_3;
LABEL_6:
    v13 = v14[0];
    *(_BYTE *)(a1 + 16) |= 1u;
    *(_DWORD *)a1 = v13;
    *(_QWORD *)(a1 + 8) = v14[1];
    return a1;
  }
  (*(void (__fastcall **)(_QWORD *))(v11 + 48))(v14);
  if ( (v15 & 1) != 0 )
    goto LABEL_6;
LABEL_3:
  (*(void (__fastcall **)(__int64, _QWORD, __int64, __int64, _QWORD, _QWORD))(*(_QWORD *)v14[0] + 32LL))(
    a1,
    v14[0],
    a3,
    a4,
    a5,
    a6);
  if ( (v15 & 1) == 0 && v14[0] )
    (*(void (__fastcall **)(_QWORD))(*(_QWORD *)v14[0] + 8LL))(v14[0]);
  return a1;
}
