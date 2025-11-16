// Function: sub_226F1F0
// Address: 0x226f1f0
//
_QWORD *__fastcall sub_226F1F0(_QWORD *a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r15
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rbx
  _QWORD *v7; // rax
  __int64 v9; // [rsp+0h] [rbp-B0h]
  __int64 v10; // [rsp+8h] [rbp-A8h]
  __int64 v11; // [rsp+10h] [rbp-A0h]
  __int64 v12; // [rsp+18h] [rbp-98h]
  __int64 v13; // [rsp+20h] [rbp-90h]
  __int64 v14; // [rsp+28h] [rbp-88h]
  _QWORD v15[16]; // [rsp+30h] [rbp-80h] BYREF

  sub_983BD0((__int64)v15, a2 + 8, a3);
  v12 = v15[6];
  v3 = v15[0];
  v4 = v15[1];
  v14 = v15[3];
  v13 = v15[5];
  v5 = v15[2];
  v11 = v15[7];
  v6 = v15[4];
  v10 = v15[8];
  v9 = v15[9];
  v7 = (_QWORD *)sub_22077B0(0x58u);
  if ( v7 )
  {
    v7[1] = v3;
    v7[2] = v4;
    v7[3] = v5;
    *v7 = &unk_4A089A0;
    v7[4] = v14;
    v7[5] = v6;
    v7[6] = v13;
    v7[7] = v12;
    v7[8] = v11;
    v7[9] = v10;
    v7[10] = v9;
  }
  *a1 = v7;
  return a1;
}
