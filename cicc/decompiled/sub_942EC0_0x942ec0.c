// Function: sub_942EC0
// Address: 0x942ec0
//
__int64 __fastcall sub_942EC0(__int64 a1, __int64 a2)
{
  __int64 v2; // rbx
  bool v3; // zf
  _BYTE *v4; // rsi
  __int64 v5; // r14
  __int64 v6; // rax
  unsigned __int64 v7; // rdx
  __int64 v8; // r14
  _QWORD *v9; // rbx
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rdx
  __int64 v13; // r12
  __int64 v14; // rsi
  __int64 v15; // r12
  __int64 v17; // rbx
  __int64 v18; // rax
  unsigned __int64 v19; // rdx
  __int64 v20; // [rsp+8h] [rbp-C8h]
  _BYTE *v21; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v22; // [rsp+18h] [rbp-B8h]
  _BYTE v23[176]; // [rsp+20h] [rbp-B0h] BYREF

  v2 = a2;
  v3 = *(_BYTE *)(a2 + 140) == 12;
  v21 = v23;
  v22 = 0x1000000000LL;
  if ( v3 )
  {
    do
      v2 = *(_QWORD *)(v2 + 160);
    while ( *(_BYTE *)(v2 + 140) == 12 );
  }
  v4 = *(_BYTE **)(v2 + 160);
  v5 = sub_941B90(a1, (__int64)v4);
  v6 = (unsigned int)v22;
  v7 = (unsigned int)v22 + 1LL;
  if ( v7 > HIDWORD(v22) )
  {
    v4 = v23;
    sub_C8D5F0(&v21, v23, v7, 8);
    v6 = (unsigned int)v22;
  }
  *(_QWORD *)&v21[8 * v6] = v5;
  v8 = *(_QWORD *)(v2 + 168);
  LODWORD(v22) = v22 + 1;
  v9 = *(_QWORD **)v8;
  if ( *(_QWORD *)v8 )
  {
    do
    {
      v4 = (_BYTE *)v9[1];
      v10 = sub_941B90(a1, (__int64)v4);
      v11 = (unsigned int)v22;
      if ( (unsigned __int64)(unsigned int)v22 + 1 > HIDWORD(v22) )
      {
        v4 = v23;
        v20 = v10;
        sub_C8D5F0(&v21, v23, (unsigned int)v22 + 1LL, 8);
        v11 = (unsigned int)v22;
        v10 = v20;
      }
      *(_QWORD *)&v21[8 * v11] = v10;
      LODWORD(v22) = v22 + 1;
      v9 = (_QWORD *)*v9;
    }
    while ( v9 );
  }
  v12 = (unsigned int)v22;
  v13 = a1 + 16;
  if ( (*(_BYTE *)(v8 + 16) & 1) != 0 )
  {
    v17 = sub_ADCD60(v13, v4, (unsigned int)v22);
    v18 = (unsigned int)v22;
    v19 = (unsigned int)v22 + 1LL;
    if ( v19 > HIDWORD(v22) )
    {
      sub_C8D5F0(&v21, v23, v19, 8);
      v18 = (unsigned int)v22;
    }
    *(_QWORD *)&v21[8 * v18] = v17;
    v12 = (unsigned int)(v22 + 1);
    LODWORD(v22) = v22 + 1;
  }
  v14 = sub_ADD430(v13, v21, v12);
  v15 = sub_ADCD40(v13, v14, 0, 0);
  if ( v21 != v23 )
    _libc_free(v21, v14);
  return v15;
}
