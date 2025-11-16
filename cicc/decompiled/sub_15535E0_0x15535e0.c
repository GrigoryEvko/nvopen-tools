// Function: sub_15535E0
// Address: 0x15535e0
//
__int64 __fastcall sub_15535E0(__int64 *a1, __int64 a2, char a3, __int64 a4)
{
  __int64 v5; // rbx
  __int64 v6; // rax
  _BYTE *v8; // rax
  _QWORD v9[2]; // [rsp+0h] [rbp-F0h] BYREF
  __int64 v10; // [rsp+10h] [rbp-E0h]
  __int64 v11; // [rsp+18h] [rbp-D8h]
  __int64 v12; // [rsp+20h] [rbp-D0h]
  __int64 v13; // [rsp+28h] [rbp-C8h]
  __int64 v14; // [rsp+30h] [rbp-C0h]
  __int64 v15; // [rsp+38h] [rbp-B8h]
  __int64 v16; // [rsp+40h] [rbp-B0h]
  __int64 v17; // [rsp+48h] [rbp-A8h]
  __int64 v18; // [rsp+50h] [rbp-A0h]
  __int64 v19; // [rsp+58h] [rbp-98h]
  __int64 v20; // [rsp+60h] [rbp-90h]
  __int64 v21; // [rsp+68h] [rbp-88h]
  __int64 v22; // [rsp+70h] [rbp-80h]
  __int64 v23; // [rsp+78h] [rbp-78h]
  char v24; // [rsp+80h] [rbp-70h]
  __int64 v25; // [rsp+88h] [rbp-68h]
  __int64 v26; // [rsp+90h] [rbp-60h]
  __int64 v27; // [rsp+98h] [rbp-58h]
  int v28; // [rsp+A0h] [rbp-50h]
  __int64 v29; // [rsp+A8h] [rbp-48h]
  __int64 v30; // [rsp+B0h] [rbp-40h]
  __int64 v31; // [rsp+B8h] [rbp-38h]

  v5 = *(_QWORD *)(a4 + 16);
  v24 = 0;
  v9[1] = 0;
  v9[0] = v5;
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
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v25 = 0;
  v26 = 0;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = 0;
  v31 = 0;
  if ( a3 )
  {
    sub_154DAA0((__int64)v9, *a1, a2);
    v8 = *(_BYTE **)(a2 + 24);
    if ( (unsigned __int64)v8 >= *(_QWORD *)(a2 + 16) )
    {
      sub_16E7DE0(a2, 32);
    }
    else
    {
      *(_QWORD *)(a2 + 24) = v8 + 1;
      *v8 = 32;
    }
    v5 = *(_QWORD *)(a4 + 16);
  }
  v6 = sub_154BC70(a4);
  sub_1550E20(a2, (__int64)a1, (__int64)v9, v6, v5);
  if ( v29 )
    j_j___libc_free_0(v29, v31 - v29);
  j___libc_free_0(v26);
  if ( v21 )
    j_j___libc_free_0(v21, v23 - v21);
  j___libc_free_0(v18);
  j___libc_free_0(v14);
  return j___libc_free_0(v10);
}
