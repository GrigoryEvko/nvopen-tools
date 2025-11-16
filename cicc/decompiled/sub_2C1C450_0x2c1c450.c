// Function: sub_2C1C450
// Address: 0x2c1c450
//
void __fastcall sub_2C1C450(__int64 a1, __int64 a2)
{
  unsigned int v4; // r15d
  __int64 *v5; // r13
  __int64 v6; // r14
  char v7; // dl
  __int64 v8; // rsi
  __int64 (*v9)(); // rax
  __int64 v10; // rax
  __int64 v11; // r8
  __int64 v12; // r9
  __int64 v13; // rdx
  _BYTE *v14; // r14
  __int64 v15; // rdi
  unsigned __int64 *v16; // r11
  __int64 v17; // r9
  __int64 v18; // r10
  unsigned __int64 v19; // rsi
  unsigned int **v20; // rdi
  __int64 v21; // r15
  __int64 v22; // rbx
  unsigned __int64 *v23; // r12
  unsigned __int64 v24; // rdi
  char v25; // al
  __int64 v26; // [rsp+8h] [rbp-108h]
  __int64 i; // [rsp+28h] [rbp-E8h]
  char v28[32]; // [rsp+30h] [rbp-E0h] BYREF
  __int16 v29; // [rsp+50h] [rbp-C0h]
  _BYTE *v30; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v31; // [rsp+68h] [rbp-A8h]
  _BYTE v32[32]; // [rsp+70h] [rbp-A0h] BYREF
  unsigned __int64 *v33; // [rsp+90h] [rbp-80h] BYREF
  __int64 v34; // [rsp+98h] [rbp-78h]
  _BYTE v35[112]; // [rsp+A0h] [rbp-70h] BYREF

  v33 = *(unsigned __int64 **)(a1 + 88);
  if ( v33 )
    sub_2AAAFA0((__int64 *)&v33);
  v4 = 1;
  sub_2BF1A90(a2, (__int64)&v33);
  sub_9C6650(&v33);
  v5 = *(__int64 **)(a1 + 48);
  v6 = *(_QWORD *)(*(_QWORD *)(a1 + 160) + 24LL);
  v30 = v32;
  v31 = 0x400000000LL;
  for ( i = (__int64)&v5[*(unsigned int *)(a1 + 56) - 1]; (__int64 *)i != v5; LODWORD(v31) = v31 + 1 )
  {
    if ( (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(*(_QWORD *)(v6 + 16) + 8LL * v4) + 8LL) - 17 <= 1 )
    {
      v7 = 0;
      v8 = *v5;
      v9 = *(__int64 (**)())(*(_QWORD *)(a1 + 40) + 24LL);
      if ( v9 != sub_2AA7510 )
      {
        v25 = ((__int64 (__fastcall *)(__int64, __int64, _QWORD))v9)(a1 + 40, v8, 0);
        v8 = *v5;
        v7 = v25;
      }
      v10 = sub_2BFB640(a2, v8, v7);
    }
    else
    {
      BYTE4(v33) = 0;
      LODWORD(v33) = 0;
      v10 = sub_2BFB120(a2, *v5, (unsigned int *)&v33);
    }
    v13 = (unsigned int)v31;
    if ( (unsigned __int64)(unsigned int)v31 + 1 > HIDWORD(v31) )
    {
      v26 = v10;
      sub_C8D5F0((__int64)&v30, v32, (unsigned int)v31 + 1LL, 8u, v11, v12);
      v13 = (unsigned int)v31;
      v10 = v26;
    }
    ++v5;
    ++v4;
    *(_QWORD *)&v30[8 * v13] = v10;
  }
  v14 = *(_BYTE **)(a1 + 136);
  if ( v14 )
  {
    v15 = *(_QWORD *)(a1 + 136);
    v33 = (unsigned __int64 *)v35;
    v34 = 0x100000000LL;
    sub_B56970(v15, (__int64)&v33);
    v16 = v33;
    v17 = (unsigned int)v34;
  }
  else
  {
    v17 = 0;
    v33 = (unsigned __int64 *)v35;
    v16 = (unsigned __int64 *)v35;
    v34 = 0x100000000LL;
  }
  v18 = *(_QWORD *)(a1 + 160);
  v19 = 0;
  v20 = *(unsigned int ***)(a2 + 904);
  v29 = 257;
  if ( v18 )
    v19 = *(_QWORD *)(v18 + 24);
  v21 = sub_B33530(v20, v19, v18, (int)v30, v31, (__int64)v28, (__int64)v16, v17, 0);
  sub_2AAF930(a1, (unsigned __int8 *)v21);
  if ( *(_BYTE *)(*(_QWORD *)(v21 + 8) + 8LL) != 7 )
    sub_2BF26E0(a2, a1 + 96, v21, 0);
  sub_2BF08A0(a2, (_BYTE *)v21, v14);
  v22 = (__int64)v33;
  v23 = &v33[7 * (unsigned int)v34];
  if ( v33 != v23 )
  {
    do
    {
      v24 = *(v23 - 3);
      v23 -= 7;
      if ( v24 )
        j_j___libc_free_0(v24);
      if ( (unsigned __int64 *)*v23 != v23 + 2 )
        j_j___libc_free_0(*v23);
    }
    while ( (unsigned __int64 *)v22 != v23 );
    v23 = v33;
  }
  if ( v23 != (unsigned __int64 *)v35 )
    _libc_free((unsigned __int64)v23);
  if ( v30 != v32 )
    _libc_free((unsigned __int64)v30);
}
