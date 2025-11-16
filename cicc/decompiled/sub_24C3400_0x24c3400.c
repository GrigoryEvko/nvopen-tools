// Function: sub_24C3400
// Address: 0x24c3400
//
unsigned __int64 __fastcall sub_24C3400(_QWORD *a1, __int64 a2, __int64 *a3, __int64 a4)
{
  __int64 v5; // rax
  __int64 v7; // r14
  __int16 v8; // dx
  __int64 v9; // rsi
  unsigned __int8 v10; // cl
  char v11; // dl
  __int64 v12; // rax
  __int64 v13; // rsi
  __int64 v14; // rdx
  __int64 v15; // r14
  __int64 v16; // rax
  _QWORD *v17; // rax
  __int64 v18; // r9
  __int64 v19; // rbx
  __int64 v20; // rsi
  __int64 v21; // r14
  unsigned int *v22; // r14
  __int64 v23; // rdx
  _BYTE *v24; // rax
  unsigned __int8 *v25; // r14
  __int64 v26; // [rsp+0h] [rbp-140h]
  unsigned int v28; // [rsp+18h] [rbp-128h]
  unsigned int *v29; // [rsp+18h] [rbp-128h]
  _BYTE v30[32]; // [rsp+20h] [rbp-120h] BYREF
  __int16 v31; // [rsp+40h] [rbp-100h]
  const char *v32[4]; // [rsp+50h] [rbp-F0h] BYREF
  __int16 v33; // [rsp+70h] [rbp-D0h]
  unsigned int *v34; // [rsp+80h] [rbp-C0h] BYREF
  unsigned int v35; // [rsp+88h] [rbp-B8h]
  char v36; // [rsp+90h] [rbp-B0h] BYREF
  __int64 v37; // [rsp+B0h] [rbp-90h]
  __int64 v38; // [rsp+B8h] [rbp-88h]
  __int64 v39; // [rsp+C0h] [rbp-80h]
  __int64 v40; // [rsp+D8h] [rbp-68h]
  void *v41; // [rsp+100h] [rbp-40h]

  if ( !*a3 )
  {
    v7 = *(_QWORD *)(a2 + 80);
    if ( v7 )
      v7 -= 24;
    v9 = sub_AA5190(v7);
    if ( v9 )
    {
      v10 = v8;
      v11 = HIBYTE(v8);
    }
    else
    {
      v11 = 0;
      v10 = 0;
    }
    v12 = v10;
    BYTE1(v12) = v11;
    v13 = sub_29F3B00(v7, v9, v12);
    if ( v13 )
      v13 -= 24;
    sub_23D0AB0((__int64)&v34, v13, 0, 0, 0);
    v14 = a1[56];
    v15 = a1[59];
    v31 = 257;
    v26 = v14;
    v16 = sub_AA4E30(v37);
    v33 = 257;
    v28 = (unsigned __int8)sub_AE5020(v16, v15);
    v17 = sub_BD2C40(80, unk_3F10A14);
    v18 = v28;
    v19 = (__int64)v17;
    if ( v17 )
      sub_B4D190((__int64)v17, v15, v26, (__int64)v32, 0, v28, 0, 0);
    v20 = v19;
    (*(void (__fastcall **)(__int64, __int64, _BYTE *, __int64, __int64, __int64))(*(_QWORD *)v40 + 16LL))(
      v40,
      v19,
      v30,
      v38,
      v39,
      v18);
    v21 = 4LL * v35;
    v29 = &v34[v21];
    if ( v34 != &v34[v21] )
    {
      v22 = v34;
      do
      {
        v23 = *((_QWORD *)v22 + 1);
        v20 = *v22;
        v22 += 4;
        sub_B99FD0(v19, v20, v23);
      }
      while ( v29 != v22 );
    }
    sub_B9D8E0(v19, v20);
    v33 = 257;
    v24 = (_BYTE *)sub_AD6530(*(_QWORD *)(v19 + 8), v20);
    v25 = (unsigned __int8 *)sub_92B530(&v34, 0x21u, v19, v24, (__int64)v32);
    v33 = 259;
    v32[0] = "sancov gate cmp";
    sub_BD6B50(v25, v32);
    *a3 = (__int64)v25;
    nullsub_61();
    v41 = &unk_49DA100;
    nullsub_63();
    if ( v34 != (unsigned int *)&v36 )
      _libc_free((unsigned __int64)v34);
  }
  v34 = (unsigned int *)a1[76];
  v5 = sub_B8C2F0(&v34, 1u, 0x186A0u, 0);
  return sub_F38250(*a3, (__int64 *)(a4 + 24), 0, 0, v5, 0, 0, 0);
}
