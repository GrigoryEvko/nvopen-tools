// Function: sub_38BF800
// Address: 0x38bf800
//
__int64 __fastcall sub_38BF800(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // rax
  __int64 v4; // rdx
  __int64 v5; // rax
  _QWORD v7[2]; // [rsp+0h] [rbp-60h] BYREF
  _QWORD v8[2]; // [rsp+10h] [rbp-50h] BYREF
  _QWORD v9[2]; // [rsp+20h] [rbp-40h] BYREF
  __int16 v10; // [rsp+30h] [rbp-30h]
  _QWORD v11[2]; // [rsp+40h] [rbp-20h] BYREF
  __int16 v12; // [rsp+50h] [rbp-10h]

  v3 = *(_QWORD *)(a1 + 16);
  v7[1] = a3;
  v4 = *(_QWORD *)(v3 + 88);
  v5 = *(_QWORD *)(v3 + 80);
  v7[0] = a2;
  v8[0] = v5;
  v9[0] = v8;
  v9[1] = v7;
  v10 = 1285;
  v8[1] = v4;
  v11[0] = v9;
  v11[1] = "$parent_frame_offset";
  v12 = 770;
  return sub_38BF510(a1, (__int64)v11);
}
