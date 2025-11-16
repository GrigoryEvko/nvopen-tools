// Function: sub_234D2B0
// Address: 0x234d2b0
//
__int64 __fastcall sub_234D2B0(__int64 a1, __int64 *a2, char a3, char a4)
{
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD *v9; // rax
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v14; // [rsp+0h] [rbp-50h] BYREF
  __int64 v15; // [rsp+8h] [rbp-48h]
  __int64 v16; // [rsp+10h] [rbp-40h]
  __int64 v17; // [rsp+18h] [rbp-38h]
  __int64 v18; // [rsp+20h] [rbp-30h]

  v6 = *a2;
  *a2 = 0;
  v17 = 0;
  v14 = v6;
  v7 = a2[1];
  a2[1] = 0;
  v15 = v7;
  v8 = a2[2];
  a2[2] = 0;
  v16 = v8;
  v18 = 0;
  v9 = (_QWORD *)sub_22077B0(0x30u);
  if ( v9 )
  {
    v9[4] = 0;
    v9[5] = 0;
    *v9 = &unk_4A0C438;
    v10 = v14;
    v14 = 0;
    v9[1] = v10;
    v11 = v15;
    v15 = 0;
    v9[2] = v11;
    v12 = v16;
    v16 = 0;
    v9[3] = v12;
  }
  *(_QWORD *)a1 = v9;
  *(_BYTE *)(a1 + 8) = a3;
  *(_BYTE *)(a1 + 9) = a4;
  sub_233F7F0((__int64)&v14);
  return a1;
}
