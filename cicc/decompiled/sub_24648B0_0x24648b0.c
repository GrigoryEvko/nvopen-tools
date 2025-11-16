// Function: sub_24648B0
// Address: 0x24648b0
//
unsigned __int64 __fastcall sub_24648B0(__int64 a1, unsigned int **a2, int a3)
{
  _BYTE *v4; // r15
  __int64 v5; // rax
  _BYTE *v6; // rax
  unsigned __int64 v7; // rax
  int v9; // [rsp+8h] [rbp-68h]
  _QWORD v10[4]; // [rsp+10h] [rbp-60h] BYREF
  __int16 v11; // [rsp+30h] [rbp-40h]

  v11 = 257;
  v4 = sub_94BCF0(a2, *(_QWORD *)(*(_QWORD *)(a1 + 16) + 144LL), *(_QWORD *)(*(_QWORD *)(a1 + 16) + 80LL), (__int64)v10);
  v5 = *(_QWORD *)(a1 + 16);
  v11 = 257;
  v6 = (_BYTE *)sub_AD64C0(*(_QWORD *)(v5 + 80), a3, 0);
  v7 = sub_929C50(a2, v4, v6, (__int64)v10, 0, 0);
  v11 = 259;
  v10[0] = "_msarg_va_o";
  return sub_24633A0((__int64 *)a2, 0x30u, v7, *(__int64 ***)(*(_QWORD *)(a1 + 16) + 96LL), (__int64)v10, 0, v9, 0);
}
