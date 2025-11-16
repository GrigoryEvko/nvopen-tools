// Function: sub_2CFC9F0
// Address: 0x2cfc9f0
//
__int64 __fastcall sub_2CFC9F0(__int64 a1, __int64 a2)
{
  _QWORD *v2; // rax
  _QWORD *v3; // rdx
  char v4; // cl
  unsigned int v5; // r12d
  __int64 v6; // rax
  __int64 v8; // rsi
  _QWORD *v9; // rbx
  _QWORD *v10; // r14
  __int64 v11; // rsi
  _QWORD *v12; // rbx
  _QWORD *v13; // r14
  __int64 v14; // rax
  _QWORD v15[2]; // [rsp+8h] [rbp-D8h] BYREF
  __int64 v16; // [rsp+18h] [rbp-C8h]
  __int64 v17; // [rsp+20h] [rbp-C0h]
  void *v18; // [rsp+30h] [rbp-B0h]
  __int64 v19; // [rsp+38h] [rbp-A8h] BYREF
  __int64 v20; // [rsp+40h] [rbp-A0h]
  __int64 v21; // [rsp+48h] [rbp-98h]
  __int64 i; // [rsp+50h] [rbp-90h]
  __int64 v23; // [rsp+60h] [rbp-80h] BYREF
  _QWORD *v24; // [rsp+68h] [rbp-78h]
  __int64 v25; // [rsp+70h] [rbp-70h]
  unsigned int v26; // [rsp+78h] [rbp-68h]
  _QWORD *v27; // [rsp+88h] [rbp-58h]
  unsigned int v28; // [rsp+98h] [rbp-48h]
  char v29; // [rsp+A0h] [rbp-40h]
  int v30; // [rsp+A9h] [rbp-37h]
  __int16 v31; // [rsp+ADh] [rbp-33h]
  char v32; // [rsp+AFh] [rbp-31h]

  v30 = 0;
  v31 = 0;
  v32 = 0;
  v23 = 0;
  v26 = 128;
  v2 = (_QWORD *)sub_C7D670(6144, 8);
  v25 = 0;
  v24 = v2;
  v19 = 2;
  v18 = &unk_4A259B8;
  v20 = 0;
  v3 = v2 + 768;
  v21 = -4096;
  for ( i = 0; v3 != v2; v2 += 6 )
  {
    if ( v2 )
    {
      v4 = v19;
      v2[2] = 0;
      v2[3] = -4096;
      *v2 = &unk_4A259B8;
      v2[1] = v4 & 6;
      v2[4] = i;
    }
  }
  v5 = (unsigned __int8)qword_5014888;
  v29 = 0;
  if ( !(_BYTE)qword_5014888 || (v5 = sub_2CFAA90((__int64)&v23, a2), !v29) )
  {
    v6 = v26;
    if ( !v26 )
      goto LABEL_7;
    goto LABEL_18;
  }
  v8 = v28;
  v29 = 0;
  if ( v28 )
  {
    v9 = v27;
    v10 = &v27[2 * v28];
    do
    {
      if ( *v9 != -8192 && *v9 != -4096 )
      {
        v11 = v9[1];
        if ( v11 )
          sub_B91220((__int64)(v9 + 1), v11);
      }
      v9 += 2;
    }
    while ( v10 != v9 );
    v8 = v28;
  }
  sub_C7D6A0((__int64)v27, 16 * v8, 8);
  v6 = v26;
  if ( v26 )
  {
LABEL_18:
    v12 = v24;
    v15[0] = 2;
    v13 = &v24[6 * v6];
    v15[1] = 0;
    v16 = -4096;
    v17 = 0;
    v19 = 2;
    v20 = 0;
    v21 = -8192;
    v18 = &unk_4A259B8;
    i = 0;
    do
    {
      v14 = v12[3];
      *v12 = &unk_49DB368;
      if ( v14 != 0 && v14 != -4096 && v14 != -8192 )
        sub_BD60C0(v12 + 1);
      v12 += 6;
    }
    while ( v13 != v12 );
    v18 = &unk_49DB368;
    if ( v21 != 0 && v21 != -4096 && v21 != -8192 )
      sub_BD60C0(&v19);
    if ( v16 != 0 && v16 != -4096 && v16 != -8192 )
      sub_BD60C0(v15);
    v6 = v26;
  }
LABEL_7:
  sub_C7D6A0((__int64)v24, 48 * v6, 8);
  return v5;
}
