// Function: sub_23AD3C0
// Address: 0x23ad3c0
//
unsigned __int64 *__fastcall sub_23AD3C0(unsigned __int64 *a1, __int64 a2, unsigned __int64 a3, int a4)
{
  _QWORD *v5; // rax
  _QWORD *v6; // rax
  _QWORD *v7; // rbx
  __int64 v8; // r14
  unsigned __int64 *v9; // rsi
  _QWORD *v10; // rbx
  __int64 v11; // r14
  unsigned __int64 *v12; // rsi
  unsigned __int64 v13; // rax
  unsigned __int64 v14; // rax
  _QWORD *v16; // rax
  _QWORD *v17; // rax
  unsigned __int64 v20; // [rsp+20h] [rbp-90h] BYREF
  unsigned __int64 v21; // [rsp+28h] [rbp-88h]
  unsigned __int64 v22; // [rsp+30h] [rbp-80h]
  __int64 v23; // [rsp+38h] [rbp-78h]
  __int64 v24; // [rsp+40h] [rbp-70h]
  _QWORD *v25; // [rsp+50h] [rbp-60h] BYREF
  _QWORD *v26; // [rsp+58h] [rbp-58h]

  if ( __PAIR64__(HIDWORD(qword_5033F08), qword_5033F08) == a3 )
  {
    sub_23A37B0(a1, a2, a3, a4);
  }
  else
  {
    v22 = 0;
    v20 = 0;
    v21 = 0;
    v23 = 0;
    v24 = 0;
    v5 = (_QWORD *)sub_22077B0(0x10u);
    if ( v5 )
      *v5 = &unk_4A0CDF8;
    v25 = v5;
    sub_23A2230(&v20, (unsigned __int64 *)&v25);
    sub_23501E0((__int64 *)&v25);
    v6 = (_QWORD *)sub_22077B0(0x10u);
    if ( v6 )
      *v6 = &unk_4A0D2F8;
    v25 = v6;
    sub_23A2230(&v20, (unsigned __int64 *)&v25);
    sub_23501E0((__int64 *)&v25);
    if ( *(_BYTE *)(a2 + 192) && *(_BYTE *)(a2 + 180) )
    {
      v17 = (_QWORD *)sub_22077B0(0x10u);
      if ( v17 )
        *v17 = &unk_4A0ED78;
      v25 = v17;
      LOBYTE(v26) = 0;
      sub_23571D0(&v20, (__int64 *)&v25);
      sub_233EFE0((__int64 *)&v25);
    }
    sub_23A1280(a2, (__int64)&v20, a3);
    sub_23A7740((unsigned __int64 *)&v25, a2, a3, a4);
    v7 = v26;
    if ( v25 != v26 )
    {
      v8 = (__int64)v25;
      do
      {
        v9 = (unsigned __int64 *)v8;
        v8 += 8;
        sub_23A2230(&v20, v9);
      }
      while ( v7 != (_QWORD *)v8 );
    }
    sub_234A900((__int64)&v25);
    sub_23AC3F0((unsigned __int64 *)&v25, a2, a3, a4);
    v10 = v26;
    if ( v25 != v26 )
    {
      v11 = (__int64)v25;
      do
      {
        v12 = (unsigned __int64 *)v11;
        v11 += 8;
        sub_23A2230(&v20, v12);
      }
      while ( v10 != (_QWORD *)v11 );
    }
    sub_234A900((__int64)&v25);
    if ( *(_BYTE *)(a2 + 192) && *(_BYTE *)(a2 + 181) && *(_DWORD *)(a2 + 168) == 3 )
    {
      v16 = (_QWORD *)sub_22077B0(0x10u);
      if ( v16 )
        *v16 = &unk_4A0DF78;
      v25 = v16;
      sub_23A2230(&v20, (unsigned __int64 *)&v25);
      sub_23501E0((__int64 *)&v25);
    }
    sub_23A2610(&v20);
    if ( (a4 & 0xFFFFFFFD) == 1 )
      sub_23A2750(a2, &v20);
    *a1 = v20;
    v13 = v21;
    a1[3] = 0;
    a1[1] = v13;
    v14 = v22;
    a1[4] = 0;
    a1[2] = v14;
  }
  return a1;
}
