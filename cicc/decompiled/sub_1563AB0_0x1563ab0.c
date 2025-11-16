// Function: sub_1563AB0
// Address: 0x1563ab0
//
__int64 __fastcall sub_1563AB0(__int64 *a1, __int64 *a2, int a3, char a4)
{
  __int64 v6; // r12
  _QWORD v8[2]; // [rsp+0h] [rbp-90h] BYREF
  int v9; // [rsp+10h] [rbp-80h] BYREF
  _QWORD *v10; // [rsp+18h] [rbp-78h]
  int *v11; // [rsp+20h] [rbp-70h]
  int *v12; // [rsp+28h] [rbp-68h]
  __int64 v13; // [rsp+30h] [rbp-60h]
  __int64 v14; // [rsp+38h] [rbp-58h]
  __int64 v15; // [rsp+40h] [rbp-50h]
  __int64 v16; // [rsp+48h] [rbp-48h]
  __int64 v17; // [rsp+50h] [rbp-40h]
  __int64 v18; // [rsp+58h] [rbp-38h]

  if ( (unsigned __int8)sub_1560260(a1, a3, a4) )
    return *a1;
  v8[0] = 0;
  v9 = 0;
  v10 = 0;
  v11 = &v9;
  v12 = &v9;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  sub_15606E0(v8, a4);
  v6 = sub_15637E0(a1, a2, a3, v8);
  sub_155CC10(v10);
  return v6;
}
