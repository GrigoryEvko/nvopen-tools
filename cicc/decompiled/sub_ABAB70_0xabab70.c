// Function: sub_ABAB70
// Address: 0xabab70
//
__int64 __fastcall sub_ABAB70(__int64 a1, __int64 a2, __int64 a3)
{
  _BYTE *v5; // rbx
  _BYTE *v6; // r14
  _BYTE *v7; // r14
  unsigned int v8; // eax
  _BYTE *v9; // rbx
  _BYTE *v10; // rbx
  __int64 v11; // [rsp+20h] [rbp-E0h] BYREF
  unsigned int v12; // [rsp+28h] [rbp-D8h]
  __int64 v13[2]; // [rsp+30h] [rbp-D0h] BYREF
  __int64 v14[2]; // [rsp+40h] [rbp-C0h] BYREF
  __int64 v15; // [rsp+50h] [rbp-B0h] BYREF
  unsigned int v16; // [rsp+58h] [rbp-A8h]
  __int64 v17; // [rsp+60h] [rbp-A0h] BYREF
  unsigned int v18; // [rsp+68h] [rbp-98h]
  __int64 v19; // [rsp+70h] [rbp-90h] BYREF
  unsigned int v20; // [rsp+78h] [rbp-88h]
  __int64 v21; // [rsp+80h] [rbp-80h] BYREF
  unsigned int v22; // [rsp+88h] [rbp-78h]
  _BYTE v23[16]; // [rsp+90h] [rbp-70h] BYREF
  _BYTE v24[16]; // [rsp+A0h] [rbp-60h] BYREF
  _BYTE v25[16]; // [rsp+B0h] [rbp-50h] BYREF
  _BYTE v26[16]; // [rsp+C0h] [rbp-40h] BYREF
  _BYTE v27[48]; // [rsp+D0h] [rbp-30h] BYREF

  if ( sub_AAF7D0(a2) || sub_AAF7D0(a3) )
  {
    sub_AADB10(a1, *(_DWORD *)(a2 + 8), 0);
  }
  else
  {
    sub_AB14C0((__int64)&v11, a2);
    sub_AB13A0((__int64)v13, a2);
    sub_AB14C0((__int64)v14, a3);
    sub_AB13A0((__int64)&v15, a3);
    sub_C4A970(v23, &v11, v14);
    sub_C4A970(v24, &v11, &v15);
    sub_C4A970(v25, v13, v14);
    sub_C4A970(v26, v13, &v15);
    v5 = v24;
    v6 = v23;
    do
    {
      if ( (int)sub_C4C880(v6, v5) < 0 )
        v6 = v5;
      v5 += 16;
    }
    while ( v5 != v27 );
    v20 = *((_DWORD *)v6 + 2);
    if ( v20 > 0x40 )
      sub_C43780(&v19, v6);
    else
      v19 = *(_QWORD *)v6;
    v7 = v23;
    sub_C46A40(&v19, 1);
    v8 = v20;
    v20 = 0;
    v9 = v24;
    v22 = v8;
    v21 = v19;
    do
    {
      if ( (int)sub_C4C880(v9, v7) < 0 )
        v7 = v9;
      v9 += 16;
    }
    while ( v9 != v27 );
    v18 = *((_DWORD *)v7 + 2);
    if ( v18 > 0x40 )
      sub_C43780(&v17, v7);
    else
      v17 = *(_QWORD *)v7;
    sub_9875E0(a1, &v17, &v21);
    if ( v18 > 0x40 && v17 )
      j_j___libc_free_0_0(v17);
    if ( v22 > 0x40 && v21 )
      j_j___libc_free_0_0(v21);
    if ( v20 > 0x40 && v19 )
      j_j___libc_free_0_0(v19);
    v10 = v27;
    do
    {
      v10 -= 16;
      if ( *((_DWORD *)v10 + 2) > 0x40u && *(_QWORD *)v10 )
        j_j___libc_free_0_0(*(_QWORD *)v10);
    }
    while ( v10 != v23 );
    if ( v16 > 0x40 && v15 )
      j_j___libc_free_0_0(v15);
    sub_969240(v14);
    sub_969240(v13);
    if ( v12 > 0x40 && v11 )
      j_j___libc_free_0_0(v11);
  }
  return a1;
}
