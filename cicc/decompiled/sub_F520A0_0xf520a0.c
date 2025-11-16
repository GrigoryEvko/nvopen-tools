// Function: sub_F520A0
// Address: 0xf520a0
//
void __fastcall sub_F520A0(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 v6; // rax
  __int16 v7; // dx
  __int64 v8; // r10
  __int64 v9; // rax
  __int64 v10; // r9
  __int64 v11; // r10
  __int64 v12; // r11
  __int64 v13; // [rsp+0h] [rbp-70h]
  __int64 v14; // [rsp+8h] [rbp-68h]
  __int64 v15; // [rsp+20h] [rbp-50h]
  __int64 v16; // [rsp+28h] [rbp-48h]
  __int64 v17[7]; // [rsp+38h] [rbp-38h] BYREF

  v4 = sub_B12000(a1 + 72);
  v5 = sub_B11F60(a1 + 80);
  if ( !(unsigned __int8)sub_F4EFF0(v4, v5, (_BYTE *)a2) && (unsigned __int8)sub_F50590(*(_QWORD *)(a2 + 8), a1) )
  {
    v16 = *(_QWORD *)(a2 + 40);
    v6 = sub_AA5190(v16);
    if ( v6 )
    {
      v8 = v6;
      v15 = v6;
      v9 = (unsigned __int8)v7;
      v13 = v8;
      BYTE1(v9) = HIBYTE(v7);
      v14 = v9;
      sub_AE7AF0((__int64)v17, a1);
      v11 = v13;
      v12 = v14;
      if ( v15 == v16 + 48 )
        goto LABEL_6;
    }
    else
    {
      sub_AE7AF0((__int64)v17, a1);
      v11 = 0;
      v12 = 0;
    }
    sub_F4EE60(a3, a2, v4, v5, (__int64)v17, v10, v11, v12);
LABEL_6:
    if ( v17[0] )
      sub_B91220((__int64)v17, v17[0]);
  }
}
