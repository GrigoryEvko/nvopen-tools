// Function: sub_DCC810
// Address: 0xdcc810
//
_QWORD *__fastcall sub_DCC810(__int64 *a1, __int64 a2, __int64 a3, char a4, unsigned int a5)
{
  _QWORD *v5; // r12
  _QWORD *v6; // r14
  __int64 v8; // rax
  _QWORD *v9; // r8
  bool v10; // dl
  int v11; // eax
  int v12; // r10d
  unsigned int v13; // r8d
  _QWORD *v14; // r12
  __int64 v16; // rax
  __int64 v17; // r15
  char v18; // al
  _QWORD *v19; // [rsp+0h] [rbp-60h]
  unsigned int v20; // [rsp+8h] [rbp-58h]
  bool v21; // [rsp+8h] [rbp-58h]
  unsigned int v22; // [rsp+8h] [rbp-58h]
  _QWORD *v24; // [rsp+10h] [rbp-50h] BYREF
  __int64 v25; // [rsp+18h] [rbp-48h]
  _QWORD v26[8]; // [rsp+20h] [rbp-40h] BYREF

  v5 = (_QWORD *)a2;
  if ( a2 == a3 )
  {
    v16 = sub_D95540(a2);
    return sub_DA2C50((__int64)a1, v16, 0, 0);
  }
  v6 = (_QWORD *)a3;
  if ( *(_BYTE *)(sub_D95540(a3) + 8) == 14 )
  {
    if ( *(_BYTE *)(sub_D95540(a2) + 8) != 14 )
      return (_QWORD *)sub_D970F0((__int64)a1);
    v17 = sub_D97190((__int64)a1, a2);
    if ( v17 != sub_D97190((__int64)a1, (__int64)v6) )
      return (_QWORD *)sub_D970F0((__int64)a1);
    v5 = sub_DCB010(a1, a2);
    v6 = sub_DCB010(a1, (__int64)v6);
  }
  v8 = sub_DBB9F0((__int64)a1, (__int64)v6, 1u, 0);
  sub_AB14C0((__int64)&v24, v8);
  if ( (unsigned int)v25 <= 0x40 )
  {
    v10 = 1LL << ((unsigned __int8)v25 - 1) != (_QWORD)v24;
  }
  else
  {
    v9 = v24;
    v20 = v25 - 1;
    v10 = 1;
    if ( (v24[v20 >> 6] & (1LL << v20)) != 0 )
    {
      v19 = v24;
      v11 = sub_C44590((__int64)&v24);
      v9 = v19;
      v10 = v20 != v11;
    }
    if ( v9 )
    {
      v21 = v10;
      j_j___libc_free_0_0(v9);
      v10 = v21;
    }
  }
  if ( (a4 & 4) == 0 )
  {
    if ( v10 )
    {
      v12 = 0;
      v13 = 4;
      goto LABEL_10;
    }
    goto LABEL_17;
  }
  v12 = 4;
  v13 = 4;
  if ( !v10 )
  {
    v18 = sub_DBED40((__int64)a1, (__int64)v5);
    v13 = 0;
    v12 = 4;
    if ( !v18 )
    {
LABEL_17:
      v12 = 0;
      v13 = 0;
    }
  }
LABEL_10:
  v22 = v12;
  v26[1] = sub_DCAF50(a1, (__int64)v6, v13);
  v26[0] = v5;
  v24 = v26;
  v25 = 0x200000002LL;
  v14 = sub_DC7EB0(a1, (__int64)&v24, v22, a5);
  if ( v24 != v26 )
    _libc_free(v24, &v24);
  return v14;
}
