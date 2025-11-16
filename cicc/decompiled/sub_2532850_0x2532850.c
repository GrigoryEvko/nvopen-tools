// Function: sub_2532850
// Address: 0x2532850
//
_QWORD *__fastcall sub_2532850(_QWORD *a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rbx
  __int64 v5; // rax
  __int64 v6; // r15
  __int64 v7; // rax
  __int64 v8; // rax
  _QWORD *v9; // r15
  _QWORD *v10; // rbx
  __int64 *v11; // r15
  __int64 *v12; // rbx
  unsigned __int64 j; // rax
  __int64 v14; // rdi
  unsigned int v15; // ecx
  __int64 v16; // rsi
  __int64 *v17; // rbx
  __int64 *v18; // r13
  __int64 v19; // rsi
  __int64 v20; // rdi
  _QWORD v24[2]; // [rsp+30h] [rbp-460h] BYREF
  char v25; // [rsp+40h] [rbp-450h]
  __int64 v26; // [rsp+50h] [rbp-440h] BYREF
  __int64 v27; // [rsp+58h] [rbp-438h]
  __int64 v28; // [rsp+60h] [rbp-430h]
  __int64 v29; // [rsp+68h] [rbp-428h]
  __int64 *v30; // [rsp+70h] [rbp-420h]
  __int64 i; // [rsp+78h] [rbp-418h]
  __int64 v32[2]; // [rsp+80h] [rbp-410h] BYREF
  __int64 *v33; // [rsp+90h] [rbp-400h]
  __int64 v34; // [rsp+98h] [rbp-3F8h]
  _BYTE v35[32]; // [rsp+A0h] [rbp-3F0h] BYREF
  __int64 *v36; // [rsp+C0h] [rbp-3D0h]
  __int64 v37; // [rsp+C8h] [rbp-3C8h]
  _QWORD v38[2]; // [rsp+D0h] [rbp-3C0h] BYREF
  _QWORD v39[50]; // [rsp+E0h] [rbp-3B0h] BYREF
  __int64 v40; // [rsp+270h] [rbp-220h] BYREF
  char *v41; // [rsp+278h] [rbp-218h]
  __int64 v42; // [rsp+280h] [rbp-210h]
  int v43; // [rsp+288h] [rbp-208h]
  char v44; // [rsp+28Ch] [rbp-204h]
  char v45; // [rsp+290h] [rbp-200h] BYREF
  _BYTE *v46; // [rsp+310h] [rbp-180h]
  __int64 v47; // [rsp+318h] [rbp-178h]
  _BYTE v48[128]; // [rsp+320h] [rbp-170h] BYREF
  _BYTE *v49; // [rsp+3A0h] [rbp-F0h]
  __int64 v50; // [rsp+3A8h] [rbp-E8h]
  _BYTE v51[128]; // [rsp+3B0h] [rbp-E0h] BYREF
  __int64 v52; // [rsp+430h] [rbp-60h]
  __int64 v53; // [rsp+438h] [rbp-58h]
  __int64 v54; // [rsp+440h] [rbp-50h]
  __int64 v55; // [rsp+448h] [rbp-48h]
  __int64 v56; // [rsp+450h] [rbp-40h]

  v4 = a3 + 24;
  v5 = sub_BC0510(a4, &unk_4F82418, a3);
  v6 = *(_QWORD *)(v4 + 8);
  v26 = 0;
  v7 = *(_QWORD *)(v5 + 8);
  v25 = 0;
  v24[1] = 0;
  v24[0] = v7;
  v27 = 0;
  v28 = 0;
  v29 = 0;
  v30 = v32;
  for ( i = 0; v4 != v6; v6 = *(_QWORD *)(v6 + 8) )
  {
    v8 = v6 - 56;
    if ( !v6 )
      v8 = 0;
    v40 = v8;
    sub_2519280((__int64)&v26, &v40);
  }
  v41 = &v45;
  v46 = v48;
  v47 = 0x1000000000LL;
  v50 = 0x1000000000LL;
  v49 = v51;
  v33 = (__int64 *)v35;
  v34 = 0x400000000LL;
  v40 = 0;
  v42 = 16;
  v43 = 0;
  v44 = 1;
  v52 = 0;
  v53 = 0;
  v54 = 0;
  v55 = 0;
  v56 = 0;
  v32[0] = 0;
  v32[1] = 0;
  v36 = v38;
  v37 = 0;
  v38[0] = 0;
  v38[1] = 1;
  sub_25112E0(v39, a3, (__int64)v24, v32, 0, 1);
  v9 = a1 + 4;
  v10 = a1 + 10;
  if ( (_DWORD)i && (unsigned __int8)sub_2532010((__int64)v39, (__int64)&v26, (__int64)&v40, 1, 1) )
  {
    memset(a1, 0, 0x60u);
    a1[1] = v9;
    *((_DWORD *)a1 + 4) = 2;
    *((_BYTE *)a1 + 28) = 1;
    a1[7] = v10;
    *((_DWORD *)a1 + 16) = 2;
    *((_BYTE *)a1 + 76) = 1;
  }
  else
  {
    a1[1] = v9;
    a1[6] = 0;
    a1[7] = v10;
    a1[8] = 2;
    *((_DWORD *)a1 + 18) = 0;
    *((_BYTE *)a1 + 76) = 1;
    a1[2] = 0x100000002LL;
    *((_DWORD *)a1 + 6) = 0;
    *((_BYTE *)a1 + 28) = 1;
    a1[4] = &qword_4F82400;
    *a1 = 1;
  }
  sub_250E960((__int64)v39);
  v11 = v33;
  v12 = &v33[(unsigned int)v34];
  if ( v33 != v12 )
  {
    for ( j = (unsigned __int64)v33; ; j = (unsigned __int64)v33 )
    {
      v14 = *v11;
      v15 = (unsigned int)((__int64)((__int64)v11 - j) >> 3) >> 7;
      v16 = 4096LL << v15;
      if ( v15 >= 0x1E )
        v16 = 0x40000000000LL;
      ++v11;
      sub_C7D6A0(v14, v16, 16);
      if ( v12 == v11 )
        break;
    }
  }
  v17 = v36;
  v18 = &v36[2 * (unsigned int)v37];
  if ( v36 != v18 )
  {
    do
    {
      v19 = v17[1];
      v20 = *v17;
      v17 += 2;
      sub_C7D6A0(v20, v19, 16);
    }
    while ( v18 != v17 );
    v18 = v36;
  }
  if ( v18 != v38 )
    _libc_free((unsigned __int64)v18);
  if ( v33 != (__int64 *)v35 )
    _libc_free((unsigned __int64)v33);
  sub_29A2B10(&v40);
  if ( v49 != v51 )
    _libc_free((unsigned __int64)v49);
  if ( v46 != v48 )
    _libc_free((unsigned __int64)v46);
  if ( !v44 )
    _libc_free((unsigned __int64)v41);
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
  sub_C7D6A0(v27, 8LL * (unsigned int)v29, 8);
  return a1;
}
