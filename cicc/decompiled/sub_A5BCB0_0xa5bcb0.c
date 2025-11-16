// Function: sub_A5BCB0
// Address: 0xa5bcb0
//
__int64 __fastcall sub_A5BCB0(__int64 a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v4; // r13
  __int64 v6; // rbx
  _BYTE *v8; // rax
  _QWORD v9[4]; // [rsp+0h] [rbp-130h] BYREF
  _QWORD v10[2]; // [rsp+20h] [rbp-110h] BYREF
  __int64 v11; // [rsp+30h] [rbp-100h]
  __int64 v12; // [rsp+38h] [rbp-F8h]
  __int64 v13; // [rsp+40h] [rbp-F0h]
  __int64 v14; // [rsp+48h] [rbp-E8h]
  __int64 v15; // [rsp+50h] [rbp-E0h]
  __int64 v16; // [rsp+58h] [rbp-D8h]
  __int64 v17; // [rsp+60h] [rbp-D0h]
  __int64 v18; // [rsp+68h] [rbp-C8h]
  __int64 v19; // [rsp+70h] [rbp-C0h]
  __int64 v20; // [rsp+78h] [rbp-B8h]
  __int64 v21; // [rsp+80h] [rbp-B0h]
  __int64 v22; // [rsp+88h] [rbp-A8h]
  __int64 v23; // [rsp+90h] [rbp-A0h]
  __int64 v24; // [rsp+98h] [rbp-98h]
  __int64 v25; // [rsp+A0h] [rbp-90h]
  __int64 v26; // [rsp+A8h] [rbp-88h]
  __int64 v27; // [rsp+B0h] [rbp-80h]
  __int64 v28; // [rsp+B8h] [rbp-78h]
  char v29; // [rsp+C0h] [rbp-70h]
  __int64 v30; // [rsp+C8h] [rbp-68h]
  __int64 v31; // [rsp+D0h] [rbp-60h]
  __int64 v32; // [rsp+D8h] [rbp-58h]
  unsigned int v33; // [rsp+E0h] [rbp-50h]
  __int64 v34; // [rsp+E8h] [rbp-48h]
  __int64 v35; // [rsp+F0h] [rbp-40h]
  __int64 v36; // [rsp+F8h] [rbp-38h]

  v4 = a2;
  v6 = *(_QWORD *)(a4 + 24);
  v29 = 0;
  v10[1] = 0;
  v10[0] = v6;
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
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v24 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v30 = 0;
  v31 = 0;
  v32 = 0;
  v33 = 0;
  v34 = 0;
  v35 = 0;
  v36 = 0;
  if ( a3 )
  {
    a2 = *(_QWORD *)(a1 + 8);
    sub_A57EC0((__int64)v10, a2, v4);
    v8 = *(_BYTE **)(v4 + 32);
    if ( (unsigned __int64)v8 >= *(_QWORD *)(v4 + 24) )
    {
      a2 = 32;
      sub_CB5D20(v4, 32);
    }
    else
    {
      *(_QWORD *)(v4 + 32) = v8 + 1;
      *v8 = 32;
    }
    v6 = *(_QWORD *)(a4 + 24);
  }
  v9[0] = off_4979428;
  v9[1] = v10;
  v9[2] = sub_A56340(a4, a2);
  v9[3] = v6;
  sub_A5A730(v4, a1, (__int64)v9);
  if ( v34 )
    j_j___libc_free_0(v34, v36 - v34);
  sub_C7D6A0(v31, 16LL * v33, 8);
  if ( v26 )
    j_j___libc_free_0(v26, v28 - v26);
  sub_C7D6A0(v23, 8LL * (unsigned int)v25, 8);
  sub_C7D6A0(v19, 8LL * (unsigned int)v21, 8);
  sub_C7D6A0(v15, 8LL * (unsigned int)v17, 8);
  return sub_C7D6A0(v11, 8LL * (unsigned int)v13, 8);
}
