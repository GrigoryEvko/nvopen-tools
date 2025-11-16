// Function: sub_B35180
// Address: 0xb35180
//
__int64 __fastcall sub_B35180(
        __int64 a1,
        __int64 a2,
        unsigned int a3,
        __int64 a4,
        unsigned __int64 a5,
        __int64 a6,
        __int64 a7)
{
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rax
  __int64 v13; // rbx
  __int64 v14; // r13
  __int64 v15; // r12
  __int64 v16; // rax
  __int64 v17; // rsi
  __int64 v18; // r12
  int v20; // [rsp+8h] [rbp-138h]
  int v21; // [rsp+10h] [rbp-130h]
  __int64 v22; // [rsp+30h] [rbp-110h]
  _QWORD v24[2]; // [rsp+40h] [rbp-100h] BYREF
  _BYTE *v25; // [rsp+50h] [rbp-F0h] BYREF
  __int64 v26; // [rsp+58h] [rbp-E8h]
  _BYTE v27[48]; // [rsp+60h] [rbp-E0h] BYREF
  _BYTE *v28; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v29; // [rsp+98h] [rbp-A8h]
  _BYTE v30[48]; // [rsp+A0h] [rbp-A0h] BYREF
  _BYTE *v31; // [rsp+D0h] [rbp-70h] BYREF
  __int64 v32; // [rsp+D8h] [rbp-68h]
  _BYTE v33[96]; // [rsp+E0h] [rbp-60h] BYREF

  v22 = sub_AA4B30(*(_QWORD *)(a1 + 48));
  v25 = v27;
  v26 = 0x400000000LL;
  sub_B6DAB0(a3, &v25, v10);
  v28 = v30;
  v11 = 0;
  v24[0] = v25;
  v24[1] = (unsigned int)v26;
  v29 = 0x600000000LL;
  v12 = 0;
  if ( a5 > 6 )
  {
    sub_C8D5F0(&v28, v30, a5, 8);
    v11 = (unsigned int)v29;
    v12 = (unsigned int)v29;
  }
  if ( a4 + 8 * a5 != a4 )
  {
    v21 = a4;
    v13 = a4;
    v20 = a5;
    v14 = a4 + 8 * a5;
    do
    {
      v15 = *(_QWORD *)(*(_QWORD *)v13 + 8LL);
      if ( v12 + 1 > (unsigned __int64)HIDWORD(v29) )
      {
        sub_C8D5F0(&v28, v30, v12 + 1, 8);
        v12 = (unsigned int)v29;
      }
      v13 += 8;
      *(_QWORD *)&v28[8 * v12] = v15;
      v12 = (unsigned int)(v29 + 1);
      LODWORD(v29) = v29 + 1;
    }
    while ( v14 != v13 );
    LODWORD(a4) = v21;
    v11 = (unsigned int)v12;
    LODWORD(a5) = v20;
  }
  v16 = sub_BCF480(a2, v28, v11, 0);
  v31 = v33;
  v32 = 0x600000000LL;
  sub_B6B020(v16, v24, &v31);
  v17 = sub_B6E160(v22, a3, v31, (unsigned int)v32);
  v18 = sub_B33A00(a1, v17, a4, a5, a7, a6, 0, 0);
  if ( v31 != v33 )
    _libc_free(v31, v17);
  if ( v28 != v30 )
    _libc_free(v28, v17);
  if ( v25 != v27 )
    _libc_free(v25, v17);
  return v18;
}
