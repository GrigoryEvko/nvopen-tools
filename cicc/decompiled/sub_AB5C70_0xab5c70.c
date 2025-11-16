// Function: sub_AB5C70
// Address: 0xab5c70
//
__int64 __fastcall sub_AB5C70(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v6; // rsi
  _BYTE *v7; // r15
  _BYTE *v8; // r15
  _BYTE *v9; // r8
  _BYTE *v10; // rdx
  int v11; // eax
  int v12; // eax
  _BYTE *v13; // rsi
  _BYTE *v14; // [rsp+8h] [rbp-108h]
  _BYTE *v15; // [rsp+10h] [rbp-100h]
  char v16; // [rsp+2Ch] [rbp-E4h] BYREF
  char v17; // [rsp+2Dh] [rbp-E3h] BYREF
  char v18; // [rsp+2Eh] [rbp-E2h] BYREF
  char v19; // [rsp+2Fh] [rbp-E1h] BYREF
  __int64 v20; // [rsp+30h] [rbp-E0h] BYREF
  unsigned int v21; // [rsp+38h] [rbp-D8h]
  __int64 v22[2]; // [rsp+40h] [rbp-D0h] BYREF
  __int64 v23[2]; // [rsp+50h] [rbp-C0h] BYREF
  __int64 v24; // [rsp+60h] [rbp-B0h] BYREF
  unsigned int v25; // [rsp+68h] [rbp-A8h]
  __int64 v26[2]; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v27; // [rsp+80h] [rbp-90h] BYREF
  int v28; // [rsp+88h] [rbp-88h]
  __int64 v29; // [rsp+90h] [rbp-80h] BYREF
  int v30; // [rsp+98h] [rbp-78h]
  _BYTE v31[16]; // [rsp+A0h] [rbp-70h] BYREF
  _BYTE v32[16]; // [rsp+B0h] [rbp-60h] BYREF
  _BYTE v33[16]; // [rsp+C0h] [rbp-50h] BYREF
  _BYTE v34[16]; // [rsp+D0h] [rbp-40h] BYREF
  _BYTE v35[48]; // [rsp+E0h] [rbp-30h] BYREF

  if ( sub_AAF7D0(a2) || sub_AAF7D0(a3) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  }
  else
  {
    sub_AB14C0((__int64)&v20, a2);
    sub_AB13A0((__int64)v22, a2);
    sub_AB14C0((__int64)v23, a3);
    v6 = a3;
    v7 = v32;
    sub_AB13A0((__int64)&v24, v6);
    sub_C4A7C0(v31, &v20, v23, &v16);
    sub_C4A7C0(v32, &v20, &v24, &v17);
    sub_C4A7C0(v33, v22, v23, &v18);
    sub_C4A7C0(v34, v22, &v24, &v19);
    if ( v16 || v17 || v18 || v19 )
    {
      sub_AADB10(a1, *(_DWORD *)(a2 + 8), 1);
    }
    else
    {
      v9 = v31;
      v10 = v32;
      do
      {
        v14 = v10;
        v15 = v9;
        v11 = sub_C4C880(v9, v10);
        v9 = v15;
        if ( v11 < 0 )
          v9 = v14;
        v10 = v14 + 16;
      }
      while ( v14 + 16 != v35 );
      sub_9865C0((__int64)&v27, (__int64)v9);
      sub_C46A40(&v27, 1);
      v12 = v28;
      v28 = 0;
      v13 = v31;
      v30 = v12;
      v29 = v27;
      do
      {
        if ( (int)sub_C4C880(v7, v13) < 0 )
          v13 = v7;
        v7 += 16;
      }
      while ( v7 != v35 );
      sub_9865C0((__int64)v26, (__int64)v13);
      sub_9875E0(a1, v26, &v29);
      sub_969240(v26);
      sub_969240(&v29);
      sub_969240(&v27);
    }
    v8 = v35;
    do
    {
      v8 -= 16;
      if ( *((_DWORD *)v8 + 2) > 0x40u && *(_QWORD *)v8 )
        j_j___libc_free_0_0(*(_QWORD *)v8);
    }
    while ( v8 != v31 );
    if ( v25 > 0x40 && v24 )
      j_j___libc_free_0_0(v24);
    sub_969240(v23);
    sub_969240(v22);
    if ( v21 > 0x40 && v20 )
      j_j___libc_free_0_0(v20);
  }
  return a1;
}
