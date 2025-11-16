// Function: sub_12A1F40
// Address: 0x12a1f40
//
__int64 __fastcall sub_12A1F40(__int64 a1, __int64 a2)
{
  __int64 v2; // r13
  bool v3; // zf
  _BYTE *v4; // rsi
  __int64 v5; // r13
  _QWORD *v6; // r15
  __int64 v7; // r8
  __int64 v8; // rax
  __int64 v9; // rdx
  __int64 v10; // r14
  __int64 v11; // rsi
  __int64 v12; // r13
  __int64 v14; // rbx
  __int64 v15; // rax
  __int64 v16; // [rsp+8h] [rbp-C8h]
  _BYTE *v17; // [rsp+10h] [rbp-C0h] BYREF
  __int64 v18; // [rsp+18h] [rbp-B8h]
  _BYTE v19[176]; // [rsp+20h] [rbp-B0h] BYREF

  v2 = a2;
  v3 = *(_BYTE *)(a2 + 140) == 12;
  v17 = v19;
  v18 = 0x1000000000LL;
  if ( v3 )
  {
    do
      v2 = *(_QWORD *)(v2 + 160);
    while ( *(_BYTE *)(v2 + 140) == 12 );
  }
  v4 = *(_BYTE **)(v2 + 160);
  *(_QWORD *)&v17[8 * (unsigned int)v18] = sub_12A0C10(a1, (__int64)v4);
  v5 = *(_QWORD *)(v2 + 168);
  LODWORD(v18) = v18 + 1;
  v6 = *(_QWORD **)v5;
  if ( *(_QWORD *)v5 )
  {
    do
    {
      v4 = (_BYTE *)v6[1];
      v7 = sub_12A0C10(a1, (__int64)v4);
      v8 = (unsigned int)v18;
      if ( (unsigned int)v18 >= HIDWORD(v18) )
      {
        v4 = v19;
        v16 = v7;
        sub_16CD150(&v17, v19, 0, 8);
        v8 = (unsigned int)v18;
        v7 = v16;
      }
      *(_QWORD *)&v17[8 * v8] = v7;
      LODWORD(v18) = v18 + 1;
      v6 = (_QWORD *)*v6;
    }
    while ( v6 );
  }
  v9 = (unsigned int)v18;
  v10 = a1 + 16;
  if ( (*(_BYTE *)(v5 + 16) & 1) != 0 )
  {
    v14 = sub_15A5DB0(a1 + 16, v4, (unsigned int)v18);
    v15 = (unsigned int)v18;
    if ( (unsigned int)v18 >= HIDWORD(v18) )
    {
      sub_16CD150(&v17, v19, 0, 8);
      v15 = (unsigned int)v18;
    }
    *(_QWORD *)&v17[8 * v15] = v14;
    v9 = (unsigned int)(v18 + 1);
    LODWORD(v18) = v18 + 1;
  }
  v11 = sub_15A66D0(v10, v17, v9);
  v12 = sub_15A5D90(v10, v11, 0, 0);
  if ( v17 != v19 )
    _libc_free(v17, v11);
  return v12;
}
