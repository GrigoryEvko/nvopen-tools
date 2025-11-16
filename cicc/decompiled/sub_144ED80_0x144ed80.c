// Function: sub_144ED80
// Address: 0x144ed80
//
_QWORD *__fastcall sub_144ED80(_QWORD *a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // rax
  __int64 v5; // rax
  __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rdi
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v15; // [rsp+0h] [rbp-230h] BYREF
  _QWORD *v16; // [rsp+8h] [rbp-228h]
  _QWORD *v17; // [rsp+10h] [rbp-220h]
  __int64 v18; // [rsp+18h] [rbp-218h]
  int v19; // [rsp+20h] [rbp-210h]
  _QWORD v20[8]; // [rsp+28h] [rbp-208h] BYREF
  __int64 v21; // [rsp+68h] [rbp-1C8h] BYREF
  __int64 v22; // [rsp+70h] [rbp-1C0h]
  __int64 v23; // [rsp+78h] [rbp-1B8h]
  _QWORD v24[16]; // [rsp+80h] [rbp-1B0h] BYREF
  char v25[8]; // [rsp+100h] [rbp-130h] BYREF
  __int64 v26; // [rsp+108h] [rbp-128h]
  unsigned __int64 v27; // [rsp+110h] [rbp-120h]
  char v28[64]; // [rsp+128h] [rbp-108h] BYREF
  __int64 v29; // [rsp+168h] [rbp-C8h]
  __int64 v30; // [rsp+170h] [rbp-C0h]
  __int64 v31; // [rsp+178h] [rbp-B8h]
  _QWORD v32[2]; // [rsp+180h] [rbp-B0h] BYREF
  unsigned __int64 v33; // [rsp+190h] [rbp-A0h]
  char v34; // [rsp+1A0h] [rbp-90h]
  char v35[64]; // [rsp+1A8h] [rbp-88h] BYREF
  __int64 v36; // [rsp+1E8h] [rbp-48h]
  __int64 v37; // [rsp+1F0h] [rbp-40h]
  __int64 v38; // [rsp+1F8h] [rbp-38h]

  sub_1444DB0(*(_QWORD **)(*(_QWORD *)a2 + 32LL), **(_QWORD **)(*(_QWORD *)a2 + 32LL) & 0xFFFFFFFFFFFFFFF8LL);
  memset(v24, 0, sizeof(v24));
  LODWORD(v24[3]) = 8;
  v24[1] = &v24[5];
  v24[2] = &v24[5];
  v20[0] = sub_1444DB0(*(_QWORD **)(*(_QWORD *)a2 + 32LL), **(_QWORD **)(*(_QWORD *)a2 + 32LL) & 0xFFFFFFFFFFFFFFF8LL);
  v32[0] = v20[0];
  v16 = v20;
  v17 = v20;
  v21 = 0;
  v22 = 0;
  v23 = 0;
  v18 = 0x100000008LL;
  v19 = 0;
  v15 = 1;
  v34 = 0;
  sub_144DD80(&v21, (__int64)v32);
  sub_16CCEE0(v32, v35, 8, v24);
  v3 = v24[13];
  memset(&v24[13], 0, 24);
  v36 = v3;
  v37 = v24[14];
  v38 = v24[15];
  sub_16CCEE0(v25, v28, 8, &v15);
  v4 = v21;
  v21 = 0;
  v29 = v4;
  v5 = v22;
  v22 = 0;
  v30 = v5;
  v6 = v23;
  v23 = 0;
  v31 = v6;
  sub_16CCEE0(a1, a1 + 5, 8, v25);
  v7 = v29;
  v29 = 0;
  a1[13] = v7;
  v8 = v30;
  v30 = 0;
  a1[14] = v8;
  v9 = v31;
  v31 = 0;
  a1[15] = v9;
  sub_16CCEE0(a1 + 16, a1 + 21, 8, v32);
  v10 = v36;
  v11 = v29;
  v36 = 0;
  a1[29] = v10;
  v12 = v37;
  v37 = 0;
  a1[30] = v12;
  v13 = v38;
  v38 = 0;
  a1[31] = v13;
  if ( v11 )
    j_j___libc_free_0(v11, v31 - v11);
  if ( v27 != v26 )
    _libc_free(v27);
  if ( v36 )
    j_j___libc_free_0(v36, v38 - v36);
  if ( v33 != v32[1] )
    _libc_free(v33);
  if ( v21 )
    j_j___libc_free_0(v21, v23 - v21);
  if ( v17 != v16 )
    _libc_free((unsigned __int64)v17);
  if ( v24[13] )
    j_j___libc_free_0(v24[13], v24[15] - v24[13]);
  if ( v24[2] != v24[1] )
    _libc_free(v24[2]);
  return a1;
}
