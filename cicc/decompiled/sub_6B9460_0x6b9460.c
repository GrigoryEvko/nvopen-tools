// Function: sub_6B9460
// Address: 0x6b9460
//
__int64 __fastcall sub_6B9460(__int64 a1, __int64 a2, __int64 a3, __int64 a4, int a5, _DWORD *a6, __int64 a7)
{
  __int64 v11; // rdx
  __int64 v12; // rcx
  __int64 v13; // r12
  char v14; // dl
  __int64 v15; // rax
  __int64 v17; // rdx
  _QWORD v18[2]; // [rsp+10h] [rbp-140h] BYREF
  unsigned int v19; // [rsp+20h] [rbp-130h]
  __int64 v20; // [rsp+24h] [rbp-12Ch]
  __int64 v21; // [rsp+30h] [rbp-120h]
  _QWORD v22[3]; // [rsp+40h] [rbp-110h] BYREF
  __int64 v23; // [rsp+58h] [rbp-F8h]
  __int64 v24; // [rsp+60h] [rbp-F0h]
  int v25; // [rsp+68h] [rbp-E8h]
  __int64 v26; // [rsp+70h] [rbp-E0h]
  char v27; // [rsp+78h] [rbp-D8h]
  _BYTE v28[18]; // [rsp+80h] [rbp-D0h] BYREF
  __int16 v29; // [rsp+92h] [rbp-BEh]

  if ( (unsigned __int8)(*(_BYTE *)(a1 + 184) - 6) <= 1u )
  {
    sub_6E3D60(v22);
    v23 = a3;
    v22[0] = a2;
    v26 = a7;
    v24 = a4;
    v25 = a5;
    sub_68A670((__int64)v22, (__int64)v18);
    sub_6E2140(5, v28, 0, 0, v22);
    v29 |= 0x2C0u;
    v13 = sub_6B8C50((__int64)v22, 0, v17);
  }
  else
  {
    sub_6E3D60(v22);
    v23 = a3;
    v22[0] = a2;
    v26 = a7;
    v24 = a4;
    v25 = a5;
    sub_68A670((__int64)v22, (__int64)v18);
    sub_6E2140(5, v28, 0, 0, v22);
    v29 |= 0x2C0u;
    v13 = sub_6B7D60((__int64)v22, 0, v11, v12);
  }
  sub_6E2B30(v22, 0);
  if ( v21 )
  {
    sub_878D40();
    sub_6E1DF0(v18[0]);
    qword_4F06BC0 = v18[1];
    unk_4F061D8 = v20;
    sub_729730(v19);
  }
  if ( v27 )
    goto LABEL_10;
  v14 = *(_BYTE *)(v13 + 140);
  if ( v14 == 12 )
  {
    v15 = v13;
    do
    {
      v15 = *(_QWORD *)(v15 + 160);
      v14 = *(_BYTE *)(v15 + 140);
    }
    while ( v14 == 12 );
  }
  if ( !v14 )
LABEL_10:
    *a6 = 1;
  return v13;
}
