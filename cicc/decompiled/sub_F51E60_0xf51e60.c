// Function: sub_F51E60
// Address: 0xf51e60
//
void __fastcall sub_F51E60(__int64 a1, __int64 a2, __int64 *a3)
{
  __int64 v3; // rdx
  __int64 v4; // r14
  __int64 v5; // r15
  __int64 v6; // rbx
  __int64 v7; // rax
  __int64 v8; // rdx
  char v9; // cl
  __int64 v10; // r9
  __int64 v11; // rax
  __int64 v12; // rdx
  __int64 v13; // [rsp+8h] [rbp-68h]
  __int64 v14; // [rsp+10h] [rbp-60h]
  __int64 v15; // [rsp+18h] [rbp-58h]
  __int64 v17[7]; // [rsp+38h] [rbp-38h] BYREF

  v3 = *(_DWORD *)(a1 + 4) & 0x7FFFFFF;
  v4 = *(_QWORD *)(*(_QWORD *)(a1 + 32 * (1 - v3)) + 24LL);
  v5 = *(_QWORD *)(*(_QWORD *)(a1 + 32 * (2 - v3)) + 24LL);
  if ( !(unsigned __int8)sub_F4EFF0(v4, v5, (_BYTE *)a2) && (unsigned __int8)sub_F506A0(*(_QWORD *)(a2 + 8), a1) )
  {
    v6 = *(_QWORD *)(a2 + 40);
    v7 = sub_AA5190(v6);
    if ( v7 )
    {
      v13 = v7;
      v9 = BYTE1(v8);
      v8 = (unsigned __int8)v8;
      BYTE1(v8) = v9;
      v14 = v7;
      v15 = v8;
      sub_AE7A80((__int64)v17, a1);
      v11 = v14;
      v12 = v15;
      if ( v13 == v6 + 48 )
        goto LABEL_6;
    }
    else
    {
      sub_AE7A80((__int64)v17, a1);
      v11 = 0;
      v12 = 0;
    }
    sub_F4EE60(a3, a2, v4, v5, (__int64)v17, v10, v11, v12);
LABEL_6:
    if ( v17[0] )
      sub_B91220((__int64)v17, v17[0]);
  }
}
