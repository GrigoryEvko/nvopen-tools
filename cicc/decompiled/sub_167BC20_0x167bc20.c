// Function: sub_167BC20
// Address: 0x167bc20
//
__int64 __fastcall sub_167BC20(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 *a5)
{
  __int64 v6; // rax
  char v7; // dl
  const char *v8; // rax
  __int64 v9; // r13
  _QWORD v11[2]; // [rsp+0h] [rbp-90h] BYREF
  const char *v12; // [rsp+10h] [rbp-80h] BYREF
  _QWORD *v13; // [rsp+18h] [rbp-78h]
  __int16 v14; // [rsp+20h] [rbp-70h]
  _QWORD v15[2]; // [rsp+30h] [rbp-60h] BYREF
  __int16 v16; // [rsp+40h] [rbp-50h]
  _BYTE v17[64]; // [rsp+50h] [rbp-40h] BYREF

  v11[0] = a3;
  v11[1] = a4;
  v6 = sub_1632000(a2, a3, a4);
  if ( !v6 )
    goto LABEL_6;
  v7 = *(_BYTE *)(v6 + 16);
  if ( v7 == 1 )
  {
    v6 = sub_164A820(*(_QWORD *)(v6 - 24));
    v7 = *(_BYTE *)(v6 + 16);
    if ( v7 == 3 )
      goto LABEL_8;
    if ( v7 )
    {
      v12 = "Linking COMDATs named '";
      v13 = v11;
      v15[0] = &v12;
      v8 = "': COMDAT key involves incomputable alias size.";
      v14 = 1283;
      goto LABEL_7;
    }
  }
  if ( v7 != 3 )
  {
LABEL_6:
    *a5 = 0;
    v12 = "Linking COMDATs named '";
    v14 = 1283;
    v13 = v11;
    v15[0] = &v12;
    v8 = "': GlobalVariable required for data dependent selection!";
LABEL_7:
    v15[1] = v8;
    v16 = 770;
    v9 = **(_QWORD **)(a1 + 8);
    sub_1670450((__int64)v17, 0, (__int64)v15);
    sub_16027F0(v9, (__int64)v17);
    return 1;
  }
LABEL_8:
  *a5 = v6;
  return 0;
}
