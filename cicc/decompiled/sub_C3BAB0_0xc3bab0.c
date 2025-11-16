// Function: sub_C3BAB0
// Address: 0xc3bab0
//
__int64 __fastcall sub_C3BAB0(__int64 a1, char a2)
{
  char v2; // al
  unsigned int v4; // r13d
  int v5; // r14d
  unsigned int v6; // eax
  unsigned __int64 v7; // rax
  unsigned __int64 v8; // rax
  __int64 v9; // rsi
  __int64 v10; // rax
  unsigned __int64 v11; // rax
  bool v12; // r14
  __int64 v14; // [rsp+0h] [rbp-60h] BYREF
  unsigned int v15; // [rsp+8h] [rbp-58h]
  _QWORD v16[2]; // [rsp+10h] [rbp-50h] BYREF
  char v17; // [rsp+24h] [rbp-3Ch]

  v2 = *(_BYTE *)(a1 + 20) & 7;
  switch ( v2 )
  {
    case 0:
      return 0;
    case 1:
      if ( (unsigned __int8)sub_C35FD0((_BYTE *)a1) )
      {
        v4 = 1;
        sub_C39170(a1);
        return v4;
      }
      return 0;
    case 3:
      return 0;
  }
  v4 = 0;
  v5 = *(_DWORD *)(a1 + 16) + 1;
  if ( v5 < (int)sub_C336A0(*(_QWORD *)a1) )
  {
    v6 = sub_C336A0(*(_QWORD *)a1);
    v7 = ((v6 | ((unsigned __int64)v6 >> 1)) >> 2) | v6 | ((unsigned __int64)v6 >> 1);
    v8 = (((v7 >> 4) | v7) >> 8) | (v7 >> 4) | v7;
    v15 = ((v8 >> 16) | v8) + 1;
    if ( v15 > 0x40 )
      sub_C43690(&v14, 1, 0);
    else
      v14 = 1;
    v9 = (unsigned int)sub_C336A0(*(_QWORD *)a1) - 1;
    if ( v15 > 0x40 )
    {
      sub_C47690(&v14, v9);
    }
    else
    {
      v10 = 0;
      if ( (_DWORD)v9 != v15 )
        v10 = v14 << v9;
      v11 = (0xFFFFFFFFFFFFFFFFLL >> -(char)v15) & v10;
      if ( !v15 )
        v11 = 0;
      v14 = v11;
    }
    sub_C37380(v16, *(_QWORD *)a1);
    sub_C36910((__int64)v16, (__int64)&v14, 0, 1);
    v12 = (*(_BYTE *)(a1 + 20) & 8) != 0;
    v17 = (8 * v12) | v17 & 0xF7;
    v4 = sub_C3ADF0(a1, (__int64)v16, a2);
    sub_C3B1F0(a1, (__int64)v16, a2);
    if ( v12 != ((*(_BYTE *)(a1 + 20) & 8) != 0) )
      sub_C34440((unsigned __int8 *)a1);
    sub_C338F0((__int64)v16);
    if ( v15 > 0x40 && v14 )
      j_j___libc_free_0_0(v14);
  }
  return v4;
}
