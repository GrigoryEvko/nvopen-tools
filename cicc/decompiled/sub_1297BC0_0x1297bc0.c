// Function: sub_1297BC0
// Address: 0x1297bc0
//
__int64 __fastcall sub_1297BC0(__int64 a1, char *a2, __int64 a3)
{
  __int64 v5; // rsi
  __int64 v6; // [rsp+8h] [rbp-88h] BYREF
  _QWORD v7[2]; // [rsp+10h] [rbp-80h] BYREF
  int v8; // [rsp+20h] [rbp-70h] BYREF
  _QWORD *v9; // [rsp+28h] [rbp-68h]
  int *v10; // [rsp+30h] [rbp-60h]
  int *v11; // [rsp+38h] [rbp-58h]
  __int64 v12; // [rsp+40h] [rbp-50h]
  __int64 v13; // [rsp+48h] [rbp-48h]
  __int64 v14; // [rsp+50h] [rbp-40h]
  __int64 v15; // [rsp+58h] [rbp-38h]
  __int64 v16; // [rsp+60h] [rbp-30h]
  __int64 v17; // [rsp+68h] [rbp-28h]

  v7[0] = 0;
  v8 = 0;
  v9 = 0;
  v10 = &v8;
  v11 = &v8;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  sub_12972A0((__int64)v7, (__int64)a2);
  if ( (a2[196] & 0x40) != 0 )
  {
    sub_15606E0(v7, 26);
    if ( !(unsigned __int8)sub_1560CB0(v7) )
      return sub_12973F0(v9);
LABEL_10:
    v5 = *(_QWORD *)(a1 + 360);
    v6 = *(_QWORD *)(a3 + 112);
    v6 = sub_15637E0(&v6, v5, 0xFFFFFFFFLL, v7);
    *(_QWORD *)(a3 + 112) = v6;
    return sub_12973F0(v9);
  }
  if ( (a2[202] & 1) != 0 || (a2[199] & 0x10) != 0 )
    sub_15606E0(v7, 3);
  if ( a2[192] < 0 )
    sub_15606E0(v7, 15);
  if ( (unsigned __int8)sub_1560CB0(v7) )
    goto LABEL_10;
  return sub_12973F0(v9);
}
