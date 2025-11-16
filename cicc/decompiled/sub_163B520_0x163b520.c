// Function: sub_163B520
// Address: 0x163b520
//
__int64 __fastcall sub_163B520(__int64 a1, __int64 *a2)
{
  __int64 v3; // rbx
  __int64 v4; // r13
  unsigned int *v5; // r15
  _BYTE *v6; // rsi
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // r12
  unsigned int *i; // [rsp+8h] [rbp-88h]
  __int64 v14; // [rsp+18h] [rbp-78h] BYREF
  __int64 *v15; // [rsp+20h] [rbp-70h] BYREF
  _BYTE *v16; // [rsp+28h] [rbp-68h]
  _BYTE *v17; // [rsp+30h] [rbp-60h]
  __int64 v18; // [rsp+40h] [rbp-50h] BYREF
  __int64 v19; // [rsp+48h] [rbp-48h]
  _QWORD *v20; // [rsp+50h] [rbp-40h]

  v15 = 0;
  v16 = 0;
  v17 = 0;
  v3 = sub_1643350(a2);
  v4 = sub_1643360(a2);
  v5 = *(unsigned int **)(a1 + 8);
  for ( i = *(unsigned int **)(a1 + 16); i != v5; v16 = v6 + 8 )
  {
    while ( 1 )
    {
      v7 = sub_15A0680(v3, *v5, 0);
      v18 = (__int64)sub_1624210(v7);
      v8 = sub_15A0680(v4, *((_QWORD *)v5 + 1), 0);
      v19 = (__int64)sub_1624210(v8);
      v9 = sub_15A0680(v3, *((_QWORD *)v5 + 2), 0);
      v20 = sub_1624210(v9);
      v10 = sub_1627350(a2, &v18, (__int64 *)3, 0, 1);
      v6 = v16;
      v14 = v10;
      if ( v16 != v17 )
        break;
      v5 += 6;
      sub_1273E00((__int64)&v15, v16, &v14);
      if ( i == v5 )
        goto LABEL_8;
    }
    if ( v16 )
    {
      *(_QWORD *)v16 = v10;
      v6 = v16;
    }
    v5 += 6;
  }
LABEL_8:
  v18 = sub_161FF10(a2, "DetailedSummary", 0xFu);
  v19 = sub_1627350(a2, v15, (__int64 *)((v16 - (_BYTE *)v15) >> 3), 0, 1);
  v11 = sub_1627350(a2, &v18, (__int64 *)2, 0, 1);
  if ( v15 )
    j_j___libc_free_0(v15, v17 - (_BYTE *)v15);
  return v11;
}
