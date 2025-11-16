// Function: sub_3039F20
// Address: 0x3039f20
//
__int64 __fastcall sub_3039F20(__int64 a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v6; // rax
  __int64 v7; // r14
  __int64 v9; // rsi
  int v10; // r15d
  __int64 v11; // rsi
  bool v12; // zf
  __int64 v13; // rax
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rdx
  __int64 v17; // r15
  __int64 v18; // r14
  __int128 v19; // rax
  int v20; // r9d
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rdx
  int v24; // r9d
  __int128 v25; // rax
  int v26; // r9d
  __int128 v27; // [rsp-20h] [rbp-B0h]
  __int128 v28; // [rsp-20h] [rbp-B0h]
  __int64 v29; // [rsp+0h] [rbp-90h]
  __int64 v30; // [rsp+8h] [rbp-88h]
  __int64 v31; // [rsp+10h] [rbp-80h] BYREF
  int v32; // [rsp+18h] [rbp-78h]
  _QWORD v33[3]; // [rsp+20h] [rbp-70h] BYREF
  int v34; // [rsp+38h] [rbp-58h]
  __int64 v35; // [rsp+40h] [rbp-50h]
  __int64 v36; // [rsp+48h] [rbp-48h]
  __int64 v37; // [rsp+50h] [rbp-40h]
  __int64 v38; // [rsp+58h] [rbp-38h]

  v6 = *(_QWORD *)(a2 + 40);
  v7 = *(_QWORD *)v6;
  if ( *(_WORD *)(*(_QWORD *)(*(_QWORD *)v6 + 48LL) + 16LL * *(unsigned int *)(v6 + 8)) != 37 )
    return a2;
  v9 = *(_QWORD *)(a2 + 80);
  v10 = *(_DWORD *)(v6 + 8);
  v31 = v9;
  if ( v9 )
  {
    sub_B96E90((__int64)&v31, v9, 1);
    v6 = *(_QWORD *)(a2 + 40);
  }
  v11 = *(_QWORD *)(v6 + 40);
  v12 = *(_DWORD *)(v11 + 24) == 51;
  v32 = *(_DWORD *)(a2 + 72);
  if ( !v12 )
  {
    v29 = *(_QWORD *)(v6 + 80);
    v30 = *(_QWORD *)(v6 + 88);
    v13 = sub_33FB310(a4, v11, *(_QWORD *)(v6 + 48), &v31, 7, 0);
    v33[1] = v14;
    v33[2] = v7;
    v34 = v10;
    v33[0] = v13;
    v15 = sub_3400BD0(a4, 8, (unsigned int)&v31, 7, 0, 0, 0);
    v17 = v16;
    v18 = v15;
    *(_QWORD *)&v19 = sub_33FB310(a4, v29, v30, &v31, 7, 0);
    *((_QWORD *)&v27 + 1) = v17;
    *(_QWORD *)&v27 = v18;
    v21 = sub_3406EB0(a4, 58, (unsigned int)&v31, 7, 0, v20, v19, v27);
    v36 = v22;
    v35 = v21;
    v37 = sub_3400BD0(a4, 8, (unsigned int)&v31, 7, 0, 0, 0);
    v38 = v23;
    *((_QWORD *)&v28 + 1) = 4;
    *(_QWORD *)&v28 = v33;
    *(_QWORD *)&v25 = sub_33FC220(a4, 536, (unsigned int)&v31, 7, 0, v24, v28);
    v7 = sub_33FAF80(
           a4,
           234,
           (unsigned int)&v31,
           **(unsigned __int16 **)(a2 + 48),
           *(_QWORD *)(*(_QWORD *)(a2 + 48) + 8LL),
           v26,
           v25);
  }
  if ( v31 )
    sub_B91220((__int64)&v31, v31);
  return v7;
}
