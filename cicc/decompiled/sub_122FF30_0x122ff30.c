// Function: sub_122FF30
// Address: 0x122ff30
//
__int64 __fastcall sub_122FF30(__int64 **a1, _QWORD *a2, __int64 *a3)
{
  unsigned __int64 v4; // rax
  unsigned int v5; // eax
  __int64 v6; // r15
  unsigned int v7; // r13d
  _QWORD *v8; // rax
  _QWORD *v9; // r12
  __int64 v10; // r13
  _QWORD *v12; // rax
  __int64 v13; // [rsp+8h] [rbp-98h]
  __int64 v14; // [rsp+10h] [rbp-90h]
  __int64 v15; // [rsp+18h] [rbp-88h]
  unsigned __int64 v16; // [rsp+20h] [rbp-80h] BYREF
  unsigned __int64 v17; // [rsp+28h] [rbp-78h] BYREF
  __int64 v18; // [rsp+30h] [rbp-70h] BYREF
  __int64 v19; // [rsp+38h] [rbp-68h] BYREF
  _QWORD v20[4]; // [rsp+40h] [rbp-60h] BYREF
  char v21; // [rsp+60h] [rbp-40h]
  char v22; // [rsp+61h] [rbp-3Fh]

  v4 = (unsigned __int64)a1[29];
  v17 = 0;
  v16 = v4;
  v5 = sub_122FE20(a1, &v18, a3);
  if ( (_BYTE)v5 )
    return 1;
  v6 = v18;
  if ( *(_BYTE *)v18 != 23 )
  {
    v10 = *(_QWORD *)(v18 + 8);
    if ( v10 == sub_BCB2A0(*a1) )
    {
      if ( !(unsigned __int8)sub_120AFE0((__int64)a1, 4, "expected ',' after branch condition")
        && !(unsigned __int8)sub_122FEA0((__int64)a1, &v19, &v16, a3)
        && !(unsigned __int8)sub_120AFE0((__int64)a1, 4, "expected ',' after true destination") )
      {
        v7 = sub_122FEA0((__int64)a1, v20, &v17, a3);
        if ( !(_BYTE)v7 )
        {
          v13 = v18;
          v14 = v20[0];
          v15 = v19;
          v12 = sub_BD2C40(72, 3u);
          v9 = v12;
          if ( v12 )
            sub_B4C9A0((__int64)v12, v15, v14, v13, 3u, v15, 0, 0);
          goto LABEL_5;
        }
      }
    }
    else
    {
      v22 = 1;
      v20[0] = "branch condition must have 'i1' type";
      v21 = 3;
      sub_11FD800((__int64)(a1 + 22), v16, (__int64)v20, 1);
    }
    return 1;
  }
  v7 = v5;
  v8 = sub_BD2C40(72, 1u);
  v9 = v8;
  if ( v8 )
    sub_B4C8F0((__int64)v8, v6, 1u, 0, 0);
LABEL_5:
  *a2 = v9;
  return v7;
}
