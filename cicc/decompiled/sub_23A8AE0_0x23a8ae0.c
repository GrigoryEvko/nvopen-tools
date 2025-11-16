// Function: sub_23A8AE0
// Address: 0x23a8ae0
//
unsigned __int64 *__fastcall sub_23A8AE0(unsigned __int64 *a1, __int64 a2, unsigned __int64 a3)
{
  _QWORD *v4; // rax
  _QWORD *v5; // rax
  _QWORD *v6; // rbx
  __int64 v7; // r14
  unsigned __int64 *v8; // rsi
  unsigned __int64 v9; // rax
  unsigned __int64 v10; // rax
  _QWORD *v12; // rax
  _QWORD *v13; // rax
  unsigned __int64 v14; // rax
  unsigned __int64 v15; // rax
  _QWORD *v16; // rax
  unsigned __int64 v18; // [rsp+10h] [rbp-90h] BYREF
  unsigned __int64 v19; // [rsp+18h] [rbp-88h]
  unsigned __int64 v20; // [rsp+20h] [rbp-80h]
  __int64 v21; // [rsp+28h] [rbp-78h]
  __int64 v22; // [rsp+30h] [rbp-70h]
  _QWORD *v23; // [rsp+40h] [rbp-60h] BYREF
  _QWORD *v24; // [rsp+48h] [rbp-58h]

  if ( __PAIR64__(HIDWORD(qword_5033F08), qword_5033F08) == a3 )
  {
    sub_23A37B0(a1, a2, a3, 1);
    return a1;
  }
  v20 = 0;
  v18 = 0;
  v19 = 0;
  v21 = 0;
  v22 = 0;
  v4 = (_QWORD *)sub_22077B0(0x10u);
  if ( v4 )
    *v4 = &unk_4A0CDF8;
  v23 = v4;
  sub_23A2230(&v18, (unsigned __int64 *)&v23);
  sub_23501E0((__int64 *)&v23);
  v5 = (_QWORD *)sub_22077B0(0x10u);
  if ( v5 )
    *v5 = &unk_4A0D2F8;
  v23 = v5;
  sub_23A2230(&v18, (unsigned __int64 *)&v23);
  sub_23501E0((__int64 *)&v23);
  if ( *(_BYTE *)(a2 + 192) && *(_BYTE *)(a2 + 180) )
  {
    v12 = (_QWORD *)sub_22077B0(0x10u);
    if ( v12 )
      *v12 = &unk_4A0ED78;
    v23 = v12;
    LOBYTE(v24) = 0;
    sub_23571D0(&v18, (__int64 *)&v23);
    sub_233EFE0((__int64 *)&v23);
  }
  sub_23A1280(a2, (__int64)&v18, a3);
  sub_23A7740((unsigned __int64 *)&v23, a2, a3, 1);
  v6 = v24;
  if ( v23 != v24 )
  {
    v7 = (__int64)v23;
    do
    {
      v8 = (unsigned __int64 *)v7;
      v7 += 8;
      sub_23A2230(&v18, v8);
    }
    while ( v6 != (_QWORD *)v7 );
  }
  sub_234A900((__int64)&v23);
  if ( qword_502E468[9] )
  {
    sub_23A2750(a2, &v18);
    v14 = v18;
    v18 = 0;
    *a1 = v14;
    v15 = v19;
    a1[3] = 0;
    a1[1] = v15;
    v19 = 0;
    a1[2] = v20;
    v20 = 0;
    a1[4] = 0;
    goto LABEL_14;
  }
  if ( byte_4FDDA08 )
  {
    v16 = (_QWORD *)sub_22077B0(0x10u);
    if ( v16 )
      *v16 = &unk_4A0DAF8;
    v23 = v16;
    sub_23A2230(&v18, (unsigned __int64 *)&v23);
    sub_23501E0((__int64 *)&v23);
    if ( !*(_BYTE *)(a2 + 192) )
      goto LABEL_13;
    goto LABEL_20;
  }
  if ( *(_BYTE *)(a2 + 192) )
  {
LABEL_20:
    if ( *(_BYTE *)(a2 + 181) && *(_DWORD *)(a2 + 168) == 3 )
    {
      v13 = (_QWORD *)sub_22077B0(0x10u);
      if ( v13 )
        *v13 = &unk_4A0DF78;
      v23 = v13;
      sub_23A2230(&v18, (unsigned __int64 *)&v23);
      sub_23501E0((__int64 *)&v23);
    }
  }
LABEL_13:
  sub_23A1080(a2, (__int64)&v18, a3, 1);
  sub_23A1110(a2, (__int64)&v18, a3, 1);
  sub_23A2610(&v18);
  sub_23A2750(a2, &v18);
  v9 = v18;
  v18 = 0;
  *a1 = v9;
  v10 = v19;
  a1[3] = 0;
  a1[1] = v10;
  v19 = 0;
  a1[2] = v20;
  v20 = 0;
  a1[4] = 0;
LABEL_14:
  sub_234A900((__int64)&v18);
  return a1;
}
