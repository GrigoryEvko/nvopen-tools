// Function: sub_20A19E0
// Address: 0x20a19e0
//
_QWORD *__fastcall sub_20A19E0(_QWORD *a1, __int64 a2, __int64 a3, _QWORD *a4)
{
  __int64 (__fastcall *v5)(__int64); // rax
  __int64 v6; // rax
  unsigned int v7; // edx
  __int64 v8; // r9
  unsigned __int8 v9; // al
  _QWORD *v10; // r12
  __int64 v12; // [rsp+0h] [rbp-30h] BYREF
  int v13; // [rsp+8h] [rbp-28h]

  v5 = *(__int64 (__fastcall **)(__int64))(*a1 + 1016LL);
  if ( v5 == sub_20A1960 )
  {
    if ( sub_20A1950((__int64)a1) && *(_QWORD *)(*(_QWORD *)(a1[1] + 608LL) + 240LL) )
      goto LABEL_4;
    return (_QWORD *)a2;
  }
  if ( ((unsigned int (*)(void))v5)() - 1 > 1 )
    return (_QWORD *)a2;
LABEL_4:
  v6 = sub_1E0A0C0(a4[4]);
  v7 = 8 * sub_15A9520(v6, 0);
  if ( v7 == 32 )
  {
    v9 = 5;
  }
  else if ( v7 > 0x20 )
  {
    v9 = 6;
    if ( v7 != 64 )
    {
      v9 = 0;
      if ( v7 == 128 )
        v9 = 7;
    }
  }
  else
  {
    v9 = 3;
    if ( v7 != 8 )
      v9 = 4 * (v7 == 16);
  }
  v12 = 0;
  v13 = 0;
  v10 = sub_1D2B300(a4, 0x13u, (__int64)&v12, v9, 0, v8);
  if ( v12 )
    sub_161E7C0((__int64)&v12, v12);
  return v10;
}
