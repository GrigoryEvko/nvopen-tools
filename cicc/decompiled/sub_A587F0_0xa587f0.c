// Function: sub_A587F0
// Address: 0xa587f0
//
__int64 __fastcall sub_A587F0(__int64 a1, __int64 a2, __int64 a3, char a4)
{
  _QWORD v6[2]; // [rsp+0h] [rbp-100h] BYREF
  __int64 v7; // [rsp+10h] [rbp-F0h]
  __int64 v8; // [rsp+18h] [rbp-E8h]
  __int64 v9; // [rsp+20h] [rbp-E0h]
  __int64 v10; // [rsp+28h] [rbp-D8h]
  __int64 v11; // [rsp+30h] [rbp-D0h]
  __int64 v12; // [rsp+38h] [rbp-C8h]
  __int64 v13; // [rsp+40h] [rbp-C0h]
  __int64 v14; // [rsp+48h] [rbp-B8h]
  __int64 v15; // [rsp+50h] [rbp-B0h]
  __int64 v16; // [rsp+58h] [rbp-A8h]
  __int64 v17; // [rsp+60h] [rbp-A0h]
  __int64 v18; // [rsp+68h] [rbp-98h]
  __int64 v19; // [rsp+70h] [rbp-90h]
  __int64 v20; // [rsp+78h] [rbp-88h]
  __int64 v21; // [rsp+80h] [rbp-80h]
  __int64 v22; // [rsp+88h] [rbp-78h]
  __int64 v23; // [rsp+90h] [rbp-70h]
  __int64 v24; // [rsp+98h] [rbp-68h]
  __int64 v25; // [rsp+A0h] [rbp-60h]
  __int64 v26; // [rsp+A8h] [rbp-58h]
  __int64 v27; // [rsp+B0h] [rbp-50h]
  __int64 v28; // [rsp+B8h] [rbp-48h]
  __int64 v29; // [rsp+C0h] [rbp-40h]
  __int64 v30; // [rsp+C8h] [rbp-38h]
  __int64 v31; // [rsp+D0h] [rbp-30h]
  __int64 v32; // [rsp+D8h] [rbp-28h]

  v21 = 0;
  v6[0] = 0;
  v6[1] = 0;
  v7 = 0;
  v8 = 0;
  v9 = 0;
  v10 = 0;
  v11 = 0;
  v12 = 0;
  v13 = 0;
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  v18 = 0;
  v19 = 0;
  v20 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  sub_A57EC0((__int64)v6, a1, a2);
  if ( !a4 && *(_BYTE *)(a1 + 8) == 15 && (*(_BYTE *)(a1 + 9) & 4) == 0 )
  {
    sub_904010(a2, " = type ");
    sub_A58460((__int64)v6, a1, a2);
  }
  if ( v30 )
    j_j___libc_free_0(v30, v32 - v30);
  sub_C7D6A0(v27, 16LL * (unsigned int)v29, 8);
  if ( v22 )
    j_j___libc_free_0(v22, v24 - v22);
  sub_C7D6A0(v19, 8LL * (unsigned int)v21, 8);
  sub_C7D6A0(v15, 8LL * (unsigned int)v17, 8);
  sub_C7D6A0(v11, 8LL * (unsigned int)v13, 8);
  return sub_C7D6A0(v7, 8LL * (unsigned int)v9, 8);
}
