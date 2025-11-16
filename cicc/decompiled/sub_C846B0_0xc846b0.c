// Function: sub_C846B0
// Address: 0xc846b0
//
void __fastcall sub_C846B0(__int64 a1, unsigned __int8 **a2)
{
  unsigned __int64 v3; // r14
  unsigned __int8 *v4; // r15
  char v5; // al
  char v6; // bl
  char *v7; // rax
  __int64 v8; // rdx
  char *v9; // rbx
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  unsigned __int8 *v12; // rax
  __int64 v13; // rdx
  unsigned __int8 *v14; // rax
  __int64 v15; // rdx
  unsigned __int8 **v16; // rsi
  __int64 v17; // [rsp+0h] [rbp-260h]
  __int64 v18; // [rsp+8h] [rbp-258h]
  unsigned __int64 v19; // [rsp+10h] [rbp-250h]
  char v20; // [rsp+20h] [rbp-240h]
  unsigned __int8 *v21; // [rsp+20h] [rbp-240h]
  __int64 v22; // [rsp+28h] [rbp-238h]
  _QWORD v23[4]; // [rsp+30h] [rbp-230h] BYREF
  __int16 v24; // [rsp+50h] [rbp-210h]
  unsigned __int8 *v25; // [rsp+60h] [rbp-200h] BYREF
  unsigned __int64 v26; // [rsp+68h] [rbp-1F8h]
  __int16 v27; // [rsp+80h] [rbp-1E0h]
  _QWORD v28[4]; // [rsp+90h] [rbp-1D0h] BYREF
  __int16 v29; // [rsp+B0h] [rbp-1B0h]
  _QWORD v30[4]; // [rsp+C0h] [rbp-1A0h] BYREF
  __int16 v31; // [rsp+E0h] [rbp-180h]
  unsigned __int8 *v32; // [rsp+F0h] [rbp-170h] BYREF
  unsigned __int64 v33; // [rsp+F8h] [rbp-168h]
  __int64 v34; // [rsp+100h] [rbp-160h]
  _BYTE v35[136]; // [rsp+108h] [rbp-158h] BYREF
  unsigned __int8 *v36; // [rsp+190h] [rbp-D0h] BYREF
  unsigned __int64 v37; // [rsp+198h] [rbp-C8h]
  __int64 v38; // [rsp+1A0h] [rbp-C0h]
  char v39; // [rsp+1A8h] [rbp-B8h] BYREF
  __int16 v40; // [rsp+1B0h] [rbp-B0h]

  v3 = (unsigned __int64)a2[1];
  v4 = *a2;
  v40 = 261;
  v36 = v4;
  v37 = v3;
  v5 = sub_C81B90((__int64)&v36, 0);
  v36 = v4;
  v6 = v5;
  v37 = v3;
  v40 = 261;
  v20 = sub_C81280((__int64)&v36, 0);
  if ( !v6 )
  {
    v33 = 0;
    v32 = v35;
    v34 = 128;
    sub_CA0EC0(a1, &v32);
    if ( v20 )
    {
      v7 = sub_C80FE0(v4, v3, 0);
      v17 = v8;
      v9 = v7;
      v10 = sub_C810B0(v32, v33, 0);
      v19 = v11;
      v18 = v10;
      v12 = sub_C80FA0(v32, v33, 0);
      v22 = v13;
      v21 = v12;
      v14 = sub_C80FA0(v4, v3, 0);
      v31 = 261;
      v25 = (unsigned __int8 *)v18;
      v30[1] = v15;
      v29 = 261;
      v27 = 261;
      v26 = v19;
      v24 = 261;
      v30[0] = v14;
      v28[0] = v21;
      v28[1] = v22;
      v23[1] = v17;
      v36 = (unsigned __int8 *)&v39;
      v37 = 0;
      v38 = 128;
      v23[0] = v9;
      sub_C81B70(&v36, (__int64)v23, (__int64)&v25, (__int64)v28, (__int64)v30);
      v16 = &v36;
      sub_C844F0(a2, &v36);
      if ( v36 != (unsigned __int8 *)&v39 )
        _libc_free(v36, &v36);
    }
    else
    {
      v40 = 257;
      v31 = 257;
      v29 = 257;
      v27 = 261;
      v25 = v4;
      v26 = v3;
      sub_C81B70(&v32, (__int64)&v25, (__int64)v28, (__int64)v30, (__int64)&v36);
      v16 = &v32;
      sub_C844F0(a2, &v32);
    }
    if ( v32 != v35 )
      _libc_free(v32, v16);
  }
}
