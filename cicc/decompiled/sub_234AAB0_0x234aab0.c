// Function: sub_234AAB0
// Address: 0x234aab0
//
__int64 __fastcall sub_234AAB0(__int64 a1, __int64 *a2, char a3)
{
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  _QWORD *v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 v10; // rdx
  __int64 v12; // [rsp+0h] [rbp-40h] BYREF
  __int64 v13; // [rsp+8h] [rbp-38h]
  __int64 v14; // [rsp+10h] [rbp-30h]
  __int64 v15; // [rsp+18h] [rbp-28h]
  __int64 v16; // [rsp+20h] [rbp-20h]

  v4 = *a2;
  *a2 = 0;
  v15 = 0;
  v12 = v4;
  v5 = a2[1];
  a2[1] = 0;
  v13 = v5;
  v6 = a2[2];
  a2[2] = 0;
  v14 = v6;
  v16 = 0;
  v7 = (_QWORD *)sub_22077B0(0x30u);
  if ( v7 )
  {
    v7[4] = 0;
    v7[5] = 0;
    *v7 = &unk_4A0C438;
    v8 = v12;
    v12 = 0;
    v7[1] = v8;
    v9 = v13;
    v13 = 0;
    v7[2] = v9;
    v10 = v14;
    v14 = 0;
    v7[3] = v10;
  }
  *(_QWORD *)a1 = v7;
  *(_BYTE *)(a1 + 8) = a3;
  sub_233F7F0((__int64)&v12);
  return a1;
}
