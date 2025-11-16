// Function: sub_3264A00
// Address: 0x3264a00
//
__int64 __fastcall sub_3264A00(__int64 *a1, _QWORD *a2)
{
  _QWORD *v2; // rcx
  __int64 *v4; // rax
  __int64 v5; // r12
  __int64 v6; // r13
  unsigned __int16 *v7; // rax
  __int64 v8; // rsi
  unsigned __int16 v9; // di
  __int64 v10; // rax
  int v11; // r15d
  __int64 v12; // rdi
  __int64 v13; // rax
  __int64 v14; // r12
  __int64 v16; // rdx
  __int64 v17; // rax
  int v18; // r9d
  __int128 v19; // [rsp-10h] [rbp-80h]
  _QWORD *v20; // [rsp+8h] [rbp-68h]
  unsigned __int16 v21; // [rsp+16h] [rbp-5Ah]
  int v22; // [rsp+18h] [rbp-58h]
  __int64 v23; // [rsp+20h] [rbp-50h] BYREF
  int v24; // [rsp+28h] [rbp-48h]
  _QWORD v25[8]; // [rsp+30h] [rbp-40h] BYREF

  v2 = a2;
  v4 = (__int64 *)a2[5];
  v5 = *v4;
  v6 = v4[1];
  v7 = (unsigned __int16 *)a2[6];
  v8 = a2[10];
  v9 = *v7;
  v10 = *((_QWORD *)v7 + 1);
  v23 = v8;
  v21 = v9;
  v11 = v9;
  v22 = v10;
  if ( v8 )
  {
    v20 = v2;
    sub_B96E90((__int64)&v23, v8, 1);
    v2 = v20;
  }
  v12 = *a1;
  v24 = *((_DWORD *)v2 + 18);
  v25[0] = v5;
  v25[1] = v6;
  v13 = sub_3402EA0(v12, 199, (unsigned int)&v23, v11, v22, 0, (__int64)v25, 1);
  if ( v13 )
  {
    v14 = v13;
  }
  else if ( (!*((_BYTE *)a1 + 33)
          || ((v16 = a1[1], v17 = 1, v21 == 1) || v21 && (v17 = v21, *(_QWORD *)(v16 + 8LL * v21 + 112)))
          && !*(_BYTE *)(v16 + 500 * v17 + 6618))
         && (unsigned __int8)sub_33DE9F0(*a1, v5, v6, 0) )
  {
    *((_QWORD *)&v19 + 1) = v6;
    *(_QWORD *)&v19 = v5;
    v14 = sub_33FAF80(*a1, 204, (unsigned int)&v23, v11, v22, v18, v19);
  }
  else
  {
    v14 = 0;
  }
  if ( v23 )
    sub_B91220((__int64)&v23, v23);
  return v14;
}
