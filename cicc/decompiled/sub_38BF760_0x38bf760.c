// Function: sub_38BF760
// Address: 0x38bf760
//
__int64 __fastcall sub_38BF760(__int64 a1, __int64 a2, __int64 a3, int a4)
{
  __int64 v4; // rax
  __int64 v5; // rdx
  __int64 v6; // rax
  _QWORD v8[2]; // [rsp+0h] [rbp-A0h] BYREF
  _QWORD v9[2]; // [rsp+10h] [rbp-90h] BYREF
  _QWORD v10[2]; // [rsp+20h] [rbp-80h] BYREF
  __int16 v11; // [rsp+30h] [rbp-70h]
  _QWORD v12[2]; // [rsp+40h] [rbp-60h] BYREF
  __int16 v13; // [rsp+50h] [rbp-50h]
  __int64 v14; // [rsp+60h] [rbp-40h]
  _QWORD v15[2]; // [rsp+80h] [rbp-20h] BYREF
  __int16 v16; // [rsp+90h] [rbp-10h]

  v4 = *(_QWORD *)(a1 + 16);
  v8[1] = a3;
  v5 = *(_QWORD *)(v4 + 88);
  v6 = *(_QWORD *)(v4 + 80);
  LODWORD(v14) = a4;
  v8[0] = a2;
  v9[0] = v6;
  v10[0] = v9;
  v10[1] = v8;
  v11 = 1285;
  v12[0] = v10;
  v12[1] = "$frame_escape_";
  v15[0] = v12;
  v9[1] = v5;
  v13 = 770;
  v15[1] = v14;
  v16 = 2306;
  return sub_38BF510(a1, (__int64)v15);
}
