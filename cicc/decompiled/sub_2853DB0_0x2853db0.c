// Function: sub_2853DB0
// Address: 0x2853db0
//
void __fastcall sub_2853DB0(__int64 *a1, __int64 a2)
{
  __int64 v3; // r13
  __int64 v4; // rsi
  unsigned __int8 *v5; // rsi
  __int64 v6; // rax
  _QWORD *v7; // r8
  _QWORD *v8; // r13
  _QWORD *v9; // r15
  _QWORD *v10; // rsi
  __int64 v11; // rax
  _QWORD *v12; // rax
  __int64 v13; // r8
  __int64 v14; // rdx
  unsigned __int64 v15; // r9
  __int64 *v16; // r13
  __int64 v17; // r12
  __int64 *v18; // rax
  __int64 v19; // r12
  _QWORD *v20; // rsi
  __int64 v21; // rax
  _QWORD *v22; // r12
  _QWORD *v23; // [rsp+8h] [rbp-98h]
  __int64 v24; // [rsp+18h] [rbp-88h]
  unsigned __int64 v25[2]; // [rsp+20h] [rbp-80h] BYREF
  __int64 v26; // [rsp+30h] [rbp-70h]
  unsigned __int8 *v27; // [rsp+40h] [rbp-60h] BYREF
  __int64 v28; // [rsp+48h] [rbp-58h]
  _BYTE v29[80]; // [rsp+50h] [rbp-50h] BYREF

  v3 = a2 + 80;
  sub_B11F20(&v27, *(_QWORD *)(*a1 + 8));
  v4 = *(_QWORD *)(a2 + 80);
  if ( v4 )
    sub_B91220(v3, v4);
  v5 = v27;
  *(_QWORD *)(a2 + 80) = v27;
  if ( v5 )
    sub_B976B0((__int64)&v27, v5, v3);
  v6 = *a1;
  v24 = a2 + 40;
  if ( *(_BYTE *)(*a1 + 16) )
  {
    v27 = v29;
    v28 = 0x300000000LL;
    v7 = *(_QWORD **)(v6 + 24);
    v8 = &v7[3 * *(unsigned int *)(v6 + 32)];
    if ( v8 == v7 )
    {
      v16 = (__int64 *)v29;
      v17 = 0;
    }
    else
    {
      v9 = *(_QWORD **)(v6 + 24);
      do
      {
        v25[0] = 4;
        v25[1] = 0;
        v26 = v9[2];
        if ( v26 != 0 && v26 != -4096 && v26 != -8192 )
          sub_BD6050(v25, *v9 & 0xFFFFFFFFFFFFFFF8LL);
        v10 = (_QWORD *)sub_B141C0(a2);
        v11 = sub_2853CD0(v25, v10);
        v12 = sub_B98A20(v11, (__int64)v10);
        v14 = (unsigned int)v28;
        v15 = (unsigned int)v28 + 1LL;
        if ( v15 > HIDWORD(v28) )
        {
          v23 = v12;
          sub_C8D5F0((__int64)&v27, v29, (unsigned int)v28 + 1LL, 8u, v13, v15);
          v14 = (unsigned int)v28;
          v12 = v23;
        }
        *(_QWORD *)&v27[8 * v14] = v12;
        LODWORD(v28) = v28 + 1;
        if ( v26 != -4096 && v26 != 0 && v26 != -8192 )
          sub_BD60C0(v25);
        v9 += 3;
      }
      while ( v8 != v9 );
      v16 = (__int64 *)v27;
      v17 = (unsigned int)v28;
    }
    v18 = (__int64 *)sub_B141C0(a2);
    v19 = sub_B00B60(v18, v16, v17);
    sub_B91340(v24, 0);
    *(_QWORD *)(a2 + 40) = v19;
    sub_B96F50(v24, 0);
    if ( v27 != v29 )
      _libc_free((unsigned __int64)v27);
  }
  else
  {
    v20 = (_QWORD *)sub_B141C0(a2);
    v21 = sub_2853CD0(*(_QWORD **)(*a1 + 24), v20);
    v22 = sub_B98A20(v21, (__int64)v20);
    sub_B91340(v24, 0);
    *(_QWORD *)(a2 + 40) = v22;
    sub_B96F50(v24, 0);
  }
}
