// Function: sub_326C490
// Address: 0x326c490
//
__int64 __fastcall sub_326C490(
        __int64 *a1,
        __int64 a2,
        __int64 a3,
        __int64 a4,
        __int64 a5,
        __int64 a6,
        __int64 a7,
        __int64 a8)
{
  __int64 result; // rax
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rsi
  __int128 *v18; // r11
  __int64 v19; // r13
  __int64 v20; // rcx
  int v21; // r10d
  __int128 v22; // [rsp-30h] [rbp-A0h]
  __int128 v23; // [rsp-10h] [rbp-80h]
  int v24; // [rsp+8h] [rbp-68h]
  __int64 v25; // [rsp+10h] [rbp-60h]
  __int64 v26; // [rsp+18h] [rbp-58h]
  int v27; // [rsp+20h] [rbp-50h]
  __int128 *v28; // [rsp+28h] [rbp-48h]
  __int64 v29; // [rsp+28h] [rbp-48h]
  __int64 v30; // [rsp+30h] [rbp-40h] BYREF
  int v31; // [rsp+38h] [rbp-38h]

  if ( !(unsigned __int8)sub_33DFCF0(a2, a3, 0) )
    return 0;
  v13 = sub_326C2C0(a7, a8, *a1, 1);
  v15 = v13;
  v16 = v14;
  if ( !v13 )
    return 0;
  v17 = *(_QWORD *)(a6 + 80);
  v18 = *(__int128 **)(a2 + 40);
  v19 = *a1;
  v20 = *(_QWORD *)(a6 + 48);
  v30 = v17;
  v21 = *(_DWORD *)(a6 + 68);
  if ( v17 )
  {
    v26 = v14;
    v24 = v20;
    v27 = *(_DWORD *)(a6 + 68);
    v28 = v18;
    v25 = v13;
    sub_B96E90((__int64)&v30, v17, 1);
    LODWORD(v20) = v24;
    v21 = v27;
    v15 = v25;
    v16 = v26;
    v18 = v28;
  }
  *((_QWORD *)&v23 + 1) = v16;
  *(_QWORD *)&v23 = v15;
  v31 = *(_DWORD *)(a6 + 72);
  *((_QWORD *)&v22 + 1) = a5;
  *(_QWORD *)&v22 = a4;
  result = sub_3412970(v19, 75, (unsigned int)&v30, v20, v21, v16, v22, *v18, v23);
  if ( v30 )
  {
    v29 = result;
    sub_B91220((__int64)&v30, v30);
    return v29;
  }
  return result;
}
