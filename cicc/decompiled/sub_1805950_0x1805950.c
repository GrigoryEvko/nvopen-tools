// Function: sub_1805950
// Address: 0x1805950
//
unsigned __int64 __fastcall sub_1805950(unsigned __int8 **a1, __int64 a2, __int64 a3)
{
  __int64 v4; // rax
  unsigned __int8 *v5; // rsi
  __int64 v6; // rax
  __int64 **v7; // rdx
  __int64 *v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rcx
  __int64 v12; // r12
  __int64 v13; // rax
  __int64 v14; // rdx
  _QWORD *v15; // rax
  _QWORD *v16; // r12
  unsigned __int64 v17; // rsi
  __int64 v18; // rax
  __int64 v19; // rsi
  __int64 v20; // rdx
  unsigned __int8 *v21; // rsi
  unsigned __int8 *v22; // rdx
  unsigned __int64 result; // rax
  __int64 v24; // rax
  unsigned __int64 *v25; // r15
  unsigned __int64 v26; // rcx
  __int64 v27; // rax
  __int64 v28; // rsi
  __int64 v29; // rdx
  unsigned __int8 *v30; // rsi
  _QWORD *v31; // [rsp+10h] [rbp-E0h]
  __int64 v33; // [rsp+18h] [rbp-D8h]
  unsigned __int64 *v34; // [rsp+18h] [rbp-D8h]
  _QWORD v35[2]; // [rsp+20h] [rbp-D0h] BYREF
  __int64 v36[2]; // [rsp+30h] [rbp-C0h] BYREF
  __int16 v37; // [rsp+40h] [rbp-B0h]
  unsigned __int8 *v38[2]; // [rsp+50h] [rbp-A0h] BYREF
  __int16 v39; // [rsp+60h] [rbp-90h]
  unsigned __int8 *v40; // [rsp+70h] [rbp-80h] BYREF
  __int64 v41; // [rsp+78h] [rbp-78h]
  unsigned __int64 *v42; // [rsp+80h] [rbp-70h]
  __int64 v43; // [rsp+88h] [rbp-68h]
  __int64 v44; // [rsp+90h] [rbp-60h]
  int v45; // [rsp+98h] [rbp-58h]
  __int64 v46; // [rsp+A0h] [rbp-50h]
  __int64 v47; // [rsp+A8h] [rbp-48h]

  v4 = sub_16498A0(a2);
  v5 = *(unsigned __int8 **)(a2 + 48);
  v40 = 0;
  v43 = v4;
  v6 = *(_QWORD *)(a2 + 40);
  v44 = 0;
  v41 = v6;
  v45 = 0;
  v46 = 0;
  v47 = 0;
  v42 = (unsigned __int64 *)(a2 + 24);
  v38[0] = v5;
  if ( v5 )
  {
    sub_1623A60((__int64)v38, (__int64)v5, 2);
    if ( v40 )
      sub_161E7C0((__int64)&v40, (__int64)v40);
    v40 = v38[0];
    if ( v38[0] )
      sub_1623210((__int64)v38, v38[0], (__int64)&v40);
  }
  v7 = (__int64 **)a1[61];
  v37 = 257;
  if ( v7 == *(__int64 ***)a3 )
  {
    v31 = (_QWORD *)a3;
  }
  else if ( *(_BYTE *)(a3 + 16) > 0x10u )
  {
    v39 = 257;
    v24 = sub_15FDBD0(45, a3, (__int64)v7, (__int64)v38, 0);
    v31 = (_QWORD *)v24;
    if ( v41 )
    {
      v25 = v42;
      sub_157E9D0(v41 + 40, v24);
      v26 = *v25;
      v27 = v31[3];
      v31[4] = v25;
      v26 &= 0xFFFFFFFFFFFFFFF8LL;
      v31[3] = v26 | v27 & 7;
      *(_QWORD *)(v26 + 8) = v31 + 3;
      *v25 = *v25 & 7 | (unsigned __int64)(v31 + 3);
    }
    sub_164B780((__int64)v31, v36);
    if ( v40 )
    {
      v35[0] = v40;
      sub_1623A60((__int64)v35, (__int64)v40, 2);
      v28 = v31[6];
      v29 = (__int64)(v31 + 6);
      if ( v28 )
      {
        sub_161E7C0((__int64)(v31 + 6), v28);
        v29 = (__int64)(v31 + 6);
      }
      v30 = (unsigned __int8 *)v35[0];
      v31[6] = v35[0];
      if ( v30 )
        sub_1623210((__int64)v35, v30, v29);
    }
  }
  else
  {
    v31 = (_QWORD *)sub_15A46C0(45, (__int64 ***)a3, v7, 0);
  }
  if ( *(_BYTE *)(a2 + 16) != 25 )
  {
    v38[0] = a1[61];
    v8 = (__int64 *)sub_15F2050(a2);
    v9 = sub_15E26F0(v8, 107, (__int64 *)v38, 1);
    v39 = 257;
    v10 = sub_1285290((__int64 *)&v40, *(_QWORD *)(v9 + 24), v9, 0, 0, (__int64)v38, 0);
    v11 = (__int64)a1[61];
    v12 = v10;
    v39 = 257;
    v37 = 257;
    v13 = sub_12AA3B0((__int64 *)&v40, 0x2Du, a3, v11, (__int64)v36);
    v31 = (_QWORD *)sub_12899C0((__int64 *)&v40, v13, v12, (__int64)v38, 0, 0);
  }
  v14 = (__int64)a1[469];
  v39 = 257;
  v37 = 257;
  v33 = v14;
  v15 = sub_1648A60(64, 1u);
  v16 = v15;
  if ( v15 )
    sub_15F9210((__int64)v15, *(_QWORD *)(*(_QWORD *)v33 + 24LL), v33, 0, 0, 0);
  if ( v41 )
  {
    v34 = v42;
    sub_157E9D0(v41 + 40, (__int64)v16);
    v17 = *v34;
    v18 = v16[3] & 7LL;
    v16[4] = v34;
    v17 &= 0xFFFFFFFFFFFFFFF8LL;
    v16[3] = v17 | v18;
    *(_QWORD *)(v17 + 8) = v16 + 3;
    *v34 = *v34 & 7 | (unsigned __int64)(v16 + 3);
  }
  sub_164B780((__int64)v16, v36);
  if ( v40 )
  {
    v35[0] = v40;
    sub_1623A60((__int64)v35, (__int64)v40, 2);
    v19 = v16[6];
    v20 = (__int64)(v16 + 6);
    if ( v19 )
    {
      sub_161E7C0((__int64)(v16 + 6), v19);
      v20 = (__int64)(v16 + 6);
    }
    v21 = (unsigned __int8 *)v35[0];
    v16[6] = v35[0];
    if ( v21 )
      sub_1623210((__int64)v35, v21, v20);
  }
  v22 = a1[394];
  v35[0] = v16;
  v35[1] = v31;
  result = sub_1285290((__int64 *)&v40, *((_QWORD *)v22 + 3), (int)v22, (int)v35, 2, (__int64)v38, 0);
  if ( v40 )
    return sub_161E7C0((__int64)&v40, (__int64)v40);
  return result;
}
