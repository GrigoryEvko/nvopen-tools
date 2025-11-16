// Function: sub_356A1B0
// Address: 0x356a1b0
//
_QWORD *__fastcall sub_356A1B0(_QWORD *a1, _QWORD *a2)
{
  unsigned __int64 v3; // rdx
  __int64 v4; // rdx
  __int64 v5; // rdx
  unsigned __int64 v6; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  unsigned __int64 v9; // rax
  __int64 v10; // rax
  __int64 v11; // rax
  unsigned __int64 v12; // rax
  unsigned __int64 v13; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  char v17[8]; // [rsp+10h] [rbp-230h] BYREF
  unsigned __int64 v18; // [rsp+18h] [rbp-228h]
  char v19; // [rsp+2Ch] [rbp-214h]
  char v20[64]; // [rsp+30h] [rbp-210h] BYREF
  unsigned __int64 v21; // [rsp+70h] [rbp-1D0h]
  __int64 v22; // [rsp+78h] [rbp-1C8h]
  __int64 v23; // [rsp+80h] [rbp-1C0h]
  char v24[8]; // [rsp+90h] [rbp-1B0h] BYREF
  unsigned __int64 v25; // [rsp+98h] [rbp-1A8h]
  char v26; // [rsp+ACh] [rbp-194h]
  char v27[64]; // [rsp+B0h] [rbp-190h] BYREF
  unsigned __int64 v28; // [rsp+F0h] [rbp-150h]
  __int64 v29; // [rsp+F8h] [rbp-148h]
  __int64 v30; // [rsp+100h] [rbp-140h]
  char v31[8]; // [rsp+110h] [rbp-130h] BYREF
  unsigned __int64 v32; // [rsp+118h] [rbp-128h]
  char v33; // [rsp+12Ch] [rbp-114h]
  _BYTE v34[64]; // [rsp+130h] [rbp-110h] BYREF
  unsigned __int64 v35; // [rsp+170h] [rbp-D0h]
  __int64 v36; // [rsp+178h] [rbp-C8h]
  __int64 v37; // [rsp+180h] [rbp-C0h]
  char v38[8]; // [rsp+190h] [rbp-B0h] BYREF
  unsigned __int64 v39; // [rsp+198h] [rbp-A8h]
  char v40; // [rsp+1ACh] [rbp-94h]
  _BYTE v41[64]; // [rsp+1B0h] [rbp-90h] BYREF
  unsigned __int64 v42; // [rsp+1F0h] [rbp-50h]
  __int64 v43; // [rsp+1F8h] [rbp-48h]
  __int64 v44; // [rsp+200h] [rbp-40h]

  sub_356A160((__int64)v24, a2);
  sub_356A0D0((__int64)v17, a2);
  sub_C8CF70((__int64)v38, v41, 8, (__int64)v27, (__int64)v24);
  v3 = v28;
  v28 = 0;
  v42 = v3;
  v4 = v29;
  v29 = 0;
  v43 = v4;
  v5 = v30;
  v30 = 0;
  v44 = v5;
  sub_C8CF70((__int64)v31, v34, 8, (__int64)v20, (__int64)v17);
  v6 = v21;
  v21 = 0;
  v35 = v6;
  v7 = v22;
  v22 = 0;
  v36 = v7;
  v8 = v23;
  v23 = 0;
  v37 = v8;
  sub_C8CF70((__int64)a1, a1 + 4, 8, (__int64)v34, (__int64)v31);
  v9 = v35;
  v35 = 0;
  a1[12] = v9;
  v10 = v36;
  v36 = 0;
  a1[13] = v10;
  v11 = v37;
  v37 = 0;
  a1[14] = v11;
  sub_C8CF70((__int64)(a1 + 15), a1 + 19, 8, (__int64)v41, (__int64)v38);
  v12 = v42;
  v13 = v35;
  v42 = 0;
  a1[27] = v12;
  v14 = v43;
  v43 = 0;
  a1[28] = v14;
  v15 = v44;
  v44 = 0;
  a1[29] = v15;
  if ( v13 )
    j_j___libc_free_0(v13);
  if ( !v33 )
    _libc_free(v32);
  if ( v42 )
    j_j___libc_free_0(v42);
  if ( !v40 )
    _libc_free(v39);
  if ( v21 )
    j_j___libc_free_0(v21);
  if ( !v19 )
    _libc_free(v18);
  if ( v28 )
    j_j___libc_free_0(v28);
  if ( !v26 )
    _libc_free(v25);
  return a1;
}
