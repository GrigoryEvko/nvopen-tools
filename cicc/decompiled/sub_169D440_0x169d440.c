// Function: sub_169D440
// Address: 0x169d440
//
__int64 __fastcall sub_169D440(__int64 a1, unsigned int a2)
{
  unsigned int v3; // eax
  unsigned __int64 v4; // rcx
  __int64 v5; // rsi
  unsigned __int64 v6; // rax
  unsigned int v7; // r14d
  bool v8; // bl
  int v10; // ebx
  unsigned __int64 v11; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v12; // [rsp+8h] [rbp-58h]
  _BYTE v13[18]; // [rsp+10h] [rbp-50h] BYREF
  char v14; // [rsp+22h] [rbp-3Eh]

  if ( (*(_BYTE *)(a1 + 18) & 7) == 3
    || (*(_BYTE *)(a1 + 18) & 6) == 0
    || (v10 = *(__int16 *)(a1 + 16), v7 = 0, v10 + 1 < (int)sub_16982D0(*(_QWORD *)a1)) )
  {
    v3 = sub_16982D0(*(_QWORD *)a1);
    v4 = (((((((((v3 | ((unsigned __int64)v3 >> 1)) >> 2) | v3 | ((unsigned __int64)v3 >> 1)) >> 4)
            | ((v3 | ((unsigned __int64)v3 >> 1)) >> 2)
            | v3
            | ((unsigned __int64)v3 >> 1)) >> 8)
          | ((((v3 | ((unsigned __int64)v3 >> 1)) >> 2) | v3 | ((unsigned __int64)v3 >> 1)) >> 4)
          | ((v3 | ((unsigned __int64)v3 >> 1)) >> 2)
          | v3
          | ((unsigned __int64)v3 >> 1)) >> 16)
        | ((((((v3 | ((unsigned __int64)v3 >> 1)) >> 2) | v3 | ((unsigned __int64)v3 >> 1)) >> 4)
          | ((v3 | ((unsigned __int64)v3 >> 1)) >> 2)
          | v3
          | ((unsigned __int64)v3 >> 1)) >> 8)
        | ((((v3 | ((unsigned __int64)v3 >> 1)) >> 2) | v3 | ((unsigned __int64)v3 >> 1)) >> 4)
        | ((v3 | ((unsigned __int64)v3 >> 1)) >> 2)
        | v3
        | ((unsigned __int64)v3 >> 1))
       + 1;
    v12 = v4;
    if ( (unsigned int)v4 <= 0x40 )
      v11 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v4) & 1;
    else
      sub_16A4EF0(&v11, 1, 0);
    v5 = (unsigned int)sub_16982D0(*(_QWORD *)a1) - 1;
    if ( v12 > 0x40 )
    {
      sub_16A7DC0(&v11, v5);
    }
    else
    {
      v6 = 0;
      if ( (_DWORD)v5 != v12 )
        v6 = (v11 << v5) & (0xFFFFFFFFFFFFFFFFLL >> -(char)v12);
      v11 = v6;
    }
    sub_1698360((__int64)v13, *(_QWORD *)a1);
    v7 = sub_169A290((__int64)v13, (__int64)&v11, 0, 0);
    v8 = (*(_BYTE *)(a1 + 18) & 8) != 0;
    v14 = (8 * v8) | v14 & 0xF7;
    if ( !v7 )
    {
      v7 = sub_169CEB0((__int16 **)a1, v13, a2);
      if ( (v7 & 0xFFFFFFEF) == 0 )
      {
        v7 = sub_169D430((__int16 **)a1, v13, a2);
        if ( v8 != ((*(_BYTE *)(a1 + 18) & 8) != 0) )
          sub_1699490(a1);
      }
    }
    sub_1698460((__int64)v13);
    if ( v12 > 0x40 && v11 )
      j_j___libc_free_0_0(v11);
  }
  return v7;
}
