// Function: sub_2354830
// Address: 0x2354830
//
__int64 __fastcall sub_2354830(unsigned __int64 *a1, __int64 *a2)
{
  __int64 v2; // rax
  _QWORD *v3; // rax
  __int64 v4; // rdx
  _QWORD *v6; // [rsp+8h] [rbp-68h] BYREF
  __int64 v7; // [rsp+10h] [rbp-60h] BYREF
  __int64 v8; // [rsp+18h] [rbp-58h]
  __int64 v9; // [rsp+20h] [rbp-50h]
  __int64 v10; // [rsp+28h] [rbp-48h]
  __int64 v11; // [rsp+30h] [rbp-40h]
  __int64 v12; // [rsp+38h] [rbp-38h]
  __int64 v13; // [rsp+40h] [rbp-30h]
  __int64 v14; // [rsp+48h] [rbp-28h]
  __int64 v15; // [rsp+50h] [rbp-20h]

  v2 = *a2;
  *a2 = 0;
  v7 = v2;
  v8 = a2[1];
  v9 = a2[2];
  v10 = a2[3];
  v11 = a2[4];
  v12 = a2[5];
  v13 = a2[6];
  v14 = a2[7];
  v15 = a2[8];
  v3 = (_QWORD *)sub_22077B0(0x50u);
  if ( v3 )
  {
    *v3 = &unk_4A0ECF8;
    v4 = v7;
    v7 = 0;
    v3[1] = v4;
    v3[2] = v8;
    v3[3] = v9;
    v3[4] = v10;
    v3[5] = v11;
    v3[6] = v12;
    v3[7] = v13;
    v3[8] = v14;
    v3[9] = v15;
  }
  v6 = v3;
  sub_2353900(a1, (unsigned __int64 *)&v6);
  if ( v6 )
    (*(void (__fastcall **)(_QWORD *))(*v6 + 8LL))(v6);
  return sub_309FA40(&v7);
}
