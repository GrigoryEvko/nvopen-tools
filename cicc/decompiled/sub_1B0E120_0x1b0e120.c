// Function: sub_1B0E120
// Address: 0x1b0e120
//
__int64 __fastcall sub_1B0E120(__int64 *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v7; // rax
  __int64 v8; // r15
  _QWORD *v10; // rax
  _QWORD *v11; // r13
  __int64 **v12; // rax
  __int64 *v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rdi
  unsigned __int64 *v16; // r12
  __int64 v17; // rax
  unsigned __int64 v18; // rcx
  __int64 v19; // rsi
  __int64 v20; // rsi
  unsigned __int8 *v21; // rsi
  __int64 *v22; // [rsp+0h] [rbp-70h]
  __int64 v23; // [rsp+8h] [rbp-68h]
  unsigned __int8 *v24; // [rsp+18h] [rbp-58h] BYREF
  char v25[16]; // [rsp+20h] [rbp-50h] BYREF
  __int16 v26; // [rsp+30h] [rbp-40h]

  v7 = sub_15A06D0(*(__int64 ***)a2, a2, (__int64)a3, a4);
  v8 = v7;
  if ( *(_BYTE *)(a2 + 16) <= 0x10u && *(_BYTE *)(v7 + 16) <= 0x10u )
    return sub_15A37B0(0x21u, (_QWORD *)a2, (_QWORD *)v7, 0);
  v26 = 257;
  v10 = sub_1648A60(56, 2u);
  v11 = v10;
  if ( v10 )
  {
    v23 = (__int64)v10;
    v12 = *(__int64 ***)a2;
    if ( *(_BYTE *)(*(_QWORD *)a2 + 8LL) == 16 )
    {
      v22 = v12[4];
      v13 = (__int64 *)sub_1643320(*v12);
      v14 = (__int64)sub_16463B0(v13, (unsigned int)v22);
    }
    else
    {
      v14 = sub_1643320(*v12);
    }
    sub_15FEC10((__int64)v11, v14, 51, 33, a2, v8, (__int64)v25, 0);
  }
  else
  {
    v23 = 0;
  }
  v15 = a1[1];
  if ( v15 )
  {
    v16 = (unsigned __int64 *)a1[2];
    sub_157E9D0(v15 + 40, (__int64)v11);
    v17 = v11[3];
    v18 = *v16;
    v11[4] = v16;
    v18 &= 0xFFFFFFFFFFFFFFF8LL;
    v11[3] = v18 | v17 & 7;
    *(_QWORD *)(v18 + 8) = v11 + 3;
    *v16 = *v16 & 7 | (unsigned __int64)(v11 + 3);
  }
  sub_164B780(v23, a3);
  v19 = *a1;
  if ( *a1 )
  {
    v24 = (unsigned __int8 *)*a1;
    sub_1623A60((__int64)&v24, v19, 2);
    v20 = v11[6];
    if ( v20 )
      sub_161E7C0((__int64)(v11 + 6), v20);
    v21 = v24;
    v11[6] = v24;
    if ( v21 )
      sub_1623210((__int64)&v24, v21, (__int64)(v11 + 6));
  }
  return (__int64)v11;
}
