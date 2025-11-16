// Function: sub_36D1870
// Address: 0x36d1870
//
_QWORD *__fastcall sub_36D1870(__int64 a1)
{
  _QWORD *v1; // r13
  __int64 v2; // rax
  __int64 v3; // r14
  bool v4; // zf
  const char *v5; // rax
  _QWORD *v6; // r12
  char v7; // al
  const char *v9; // [rsp+10h] [rbp-50h] BYREF
  char v10; // [rsp+30h] [rbp-30h]
  char v11; // [rsp+31h] [rbp-2Fh]

  v1 = (_QWORD *)sub_BCE3C0(*(__int64 **)(a1 + 8), 0);
  v2 = sub_BCE3C0(*(__int64 **)(a1 + 8), 0);
  v3 = sub_AD6530(v2, 0);
  v4 = **(_BYTE **)(a1 + 16) == 0;
  v5 = "__init_array_start";
  v11 = 1;
  if ( v4 )
    v5 = "__fini_array_start";
  v10 = 3;
  v9 = v5;
  v6 = sub_BD2C40(88, unk_3F0FAE8);
  if ( v6 )
    sub_B30000((__int64)v6, *(_QWORD *)a1, v1, 0, 4, v3, (__int64)&v9, 0, 0, 0x100000001LL, 0);
  v7 = v6[4] & 0xCF | 0x20;
  *((_BYTE *)v6 + 32) = v7;
  if ( (v7 & 0xF) != 9 )
    *((_BYTE *)v6 + 33) |= 0x40u;
  return v6;
}
