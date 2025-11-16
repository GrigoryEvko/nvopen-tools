// Function: sub_1296AD0
// Address: 0x1296ad0
//
__int64 __fastcall sub_1296AD0(__int64 *a1, _QWORD *a2)
{
  _QWORD *v4; // rbx
  _QWORD *v5; // r15
  _QWORD *v6; // r13
  __int64 v7; // rax
  _QWORD *v8; // r13
  __int64 v9; // rdi
  unsigned __int64 *v10; // rbx
  __int64 v11; // rax
  unsigned __int64 v12; // rcx
  __int64 v13; // rsi
  _QWORD *v14; // rdx
  __int64 v15; // rsi
  __int64 v17; // [rsp+8h] [rbp-68h]
  __int64 v18; // [rsp+18h] [rbp-58h] BYREF
  _BYTE v19[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v20; // [rsp+30h] [rbp-40h]

  v4 = (_QWORD *)sub_12A4D50(a1, "do.body", 0, 0);
  v5 = (_QWORD *)sub_12A4D50(a1, "do.end", 0, 0);
  sub_1290AF0(a1, v4, 0);
  v6 = (_QWORD *)sub_12A4D50(a1, "do.cond", 0, 0);
  sub_1296350(a1, a2[9]);
  sub_1290AF0(a1, v6, 0);
  sub_1290930((__int64)a1, (unsigned int *)(a2[6] + 36LL));
  sub_127C770((_QWORD *)(a2[6] + 36LL));
  v17 = sub_127FEC0((__int64)a1, a2[6]);
  v20 = 257;
  v7 = sub_1648A60(56, 3);
  v8 = (_QWORD *)v7;
  if ( v7 )
    sub_15F83E0(v7, v4, v5, v17, 0);
  v9 = a1[7];
  if ( !v9 )
  {
    sub_164B780(v8, v19);
    v13 = a1[6];
    if ( !v13 )
    {
      if ( !v8 )
        return sub_1290AF0(a1, v5, 0);
      goto LABEL_9;
    }
    goto LABEL_5;
  }
  v10 = (unsigned __int64 *)a1[8];
  sub_157E9D0(v9 + 40, v8);
  v11 = v8[3];
  v12 = *v10;
  v8[4] = v10;
  v12 &= 0xFFFFFFFFFFFFFFF8LL;
  v8[3] = v12 | v11 & 7;
  *(_QWORD *)(v12 + 8) = v8 + 3;
  *v10 = *v10 & 7 | (unsigned __int64)(v8 + 3);
  sub_164B780(v8, v19);
  v13 = a1[6];
  if ( v13 )
  {
LABEL_5:
    v18 = v13;
    sub_1623A60(&v18, v13, 2);
    v14 = v8 + 6;
    if ( v8[6] )
    {
      sub_161E7C0(v8 + 6);
      v14 = v8 + 6;
    }
    v15 = v18;
    v8[6] = v18;
    if ( v15 )
      sub_1623210(&v18, v15, v14);
  }
LABEL_9:
  if ( a2[8] )
    sub_1291160((__int64)a1, (__int64)v8, (__int64)a2);
  return sub_1290AF0(a1, v5, 0);
}
