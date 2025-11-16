// Function: sub_1C8E770
// Address: 0x1c8e770
//
__int64 __fastcall sub_1C8E770(char **a1, __int64 a2)
{
  unsigned __int8 *v3; // r14
  __int64 v4; // rax
  unsigned __int8 *v5; // rsi
  __int64 v6; // rax
  __int64 *v7; // rax
  __int64 v8; // r12
  __int64 v9; // rsi
  __int64 v10; // rdx
  __int64 v11; // r12
  char **v13; // [rsp+8h] [rbp-A8h] BYREF
  _QWORD v14[2]; // [rsp+10h] [rbp-A0h] BYREF
  unsigned __int8 *v15; // [rsp+20h] [rbp-90h] BYREF
  char *v16; // [rsp+28h] [rbp-88h]
  __int16 v17; // [rsp+30h] [rbp-80h]
  __int64 v18[5]; // [rsp+40h] [rbp-70h] BYREF
  int v19; // [rsp+68h] [rbp-48h]
  __int64 v20; // [rsp+70h] [rbp-40h]
  __int64 v21; // [rsp+78h] [rbp-38h]

  v13 = a1;
  v3 = (unsigned __int8 *)sub_1646BA0(**((__int64 ***)*a1 + 2), 0);
  v4 = sub_16498A0(a2);
  v5 = *(unsigned __int8 **)(a2 + 48);
  v18[0] = 0;
  v18[3] = v4;
  v6 = *(_QWORD *)(a2 + 40);
  v18[4] = 0;
  v18[1] = v6;
  v19 = 0;
  v20 = 0;
  v21 = 0;
  v18[2] = a2 + 24;
  v15 = v5;
  if ( v5 )
  {
    sub_1623A60((__int64)&v15, (__int64)v5, 2);
    if ( v18[0] )
      sub_161E7C0((__int64)v18, v18[0]);
    v18[0] = (__int64)v15;
    if ( v15 )
      sub_1623210((__int64)&v15, v15, (__int64)v18);
  }
  v15 = v3;
  v16 = *v13;
  v7 = (__int64 *)sub_15F2050(a2);
  v8 = sub_15E26F0(v7, 4046, (__int64 *)&v15, 2);
  v14[0] = sub_1649960((__int64)v13);
  v17 = 773;
  v15 = (unsigned __int8 *)v14;
  v16 = ".gen";
  v9 = *(_QWORD *)(v8 + 24);
  v14[1] = v10;
  v11 = sub_1285290(v18, v9, v8, (int)&v13, 1, (__int64)&v15, 0);
  if ( v18[0] )
    sub_161E7C0((__int64)v18, v18[0]);
  return v11;
}
