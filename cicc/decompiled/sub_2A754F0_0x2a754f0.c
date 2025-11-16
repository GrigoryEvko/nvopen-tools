// Function: sub_2A754F0
// Address: 0x2a754f0
//
__int64 __fastcall sub_2A754F0(__int64 a1, __int64 a2)
{
  __int64 *v4; // r13
  __int64 *v5; // rax
  __int64 v6; // rsi
  __int64 **v7; // r14
  __int64 v8; // rax
  int v9; // ecx
  __int64 v10; // rdi
  int v11; // ecx
  unsigned int v12; // edx
  __int64 *v13; // rax
  __int64 v14; // r8
  char *v15; // r15
  __int64 *v16; // r13
  __int64 *v17; // r14
  unsigned int v18; // r13d
  const char *v20; // rax
  __int64 v21; // rsi
  __int64 v22; // rdx
  __int64 v23; // rdx
  __int64 v24; // r14
  bool v25; // al
  __int64 v26; // rsi
  __int64 v27; // rdx
  __int64 v28; // rbx
  __int64 v29; // rdx
  __int64 *v30; // r12
  unsigned __int64 v31; // rcx
  __int64 v32; // r9
  int v33; // eax
  unsigned __int64 *v34; // rdi
  __int64 v35; // rax
  int v36; // eax
  __int64 v37; // rsi
  unsigned __int8 *v38; // rsi
  int v39; // r9d
  __int64 v40; // r14
  char *v41; // r12
  int v42; // edx
  unsigned __int64 *v43; // r12
  int v44; // eax
  int v45; // [rsp+0h] [rbp-80h]
  __int64 v46; // [rsp+8h] [rbp-78h]
  unsigned __int64 *v47; // [rsp+8h] [rbp-78h]
  int v48; // [rsp+8h] [rbp-78h]
  unsigned __int64 v49; // [rsp+18h] [rbp-68h] BYREF
  __int64 v50; // [rsp+20h] [rbp-60h] BYREF
  __int64 v51; // [rsp+28h] [rbp-58h]
  const char *v52; // [rsp+30h] [rbp-50h]
  __int16 v53; // [rsp+40h] [rbp-40h]

  v4 = sub_DD8400(*(_QWORD *)(a1 + 16), *(_QWORD *)(a2 - 64));
  v5 = sub_DD8400(*(_QWORD *)(a1 + 16), *(_QWORD *)(a2 - 32));
  v6 = *(_QWORD *)(a2 + 40);
  v7 = (__int64 **)v5;
  v8 = *(_QWORD *)(a1 + 8);
  v9 = *(_DWORD *)(v8 + 24);
  v10 = *(_QWORD *)(v8 + 8);
  if ( !v9 )
  {
LABEL_26:
    v15 = 0;
    goto LABEL_4;
  }
  v11 = v9 - 1;
  v12 = v11 & (((unsigned int)v6 >> 9) ^ ((unsigned int)v6 >> 4));
  v13 = (__int64 *)(v10 + 16LL * v12);
  v14 = *v13;
  if ( v6 != *v13 )
  {
    v36 = 1;
    while ( v14 != -4096 )
    {
      v39 = v36 + 1;
      v12 = v11 & (v36 + v12);
      v13 = (__int64 *)(v10 + 16LL * v12);
      v14 = *v13;
      if ( v6 == *v13 )
        goto LABEL_3;
      v36 = v39;
    }
    goto LABEL_26;
  }
LABEL_3:
  v15 = (char *)v13[1];
LABEL_4:
  v16 = sub_DDF4E0(*(_QWORD *)(a1 + 16), (__int64 **)v4, v15);
  v17 = sub_DDF4E0(*(_QWORD *)(a1 + 16), v7, v15);
  if ( !(unsigned __int8)sub_DBED40(*(_QWORD *)(a1 + 16), (__int64)v16) )
    return 0;
  v18 = sub_DBED40(*(_QWORD *)(a1 + 16), (__int64)v17);
  if ( !(_BYTE)v18 )
    return 0;
  v20 = sub_BD5D20(a2);
  v21 = *(_QWORD *)(a2 - 64);
  v51 = v22;
  v23 = *(_QWORD *)(a2 - 32);
  v53 = 773;
  v50 = (__int64)v20;
  v52 = ".udiv";
  v24 = sub_B504D0(19, v21, v23, (__int64)&v50, a2 + 24, 0);
  v25 = sub_B44E60(a2);
  sub_B448B0(v24, v25);
  sub_BD84D0(a2, v24);
  v26 = *(_QWORD *)(a2 + 48);
  v50 = v26;
  if ( v26 )
  {
    sub_B96E90((__int64)&v50, v26, 1);
    v27 = v24 + 48;
    if ( (__int64 *)(v24 + 48) == &v50 )
    {
      if ( v50 )
        sub_B91220((__int64)&v50, v50);
      goto LABEL_12;
    }
    v37 = *(_QWORD *)(v24 + 48);
    if ( !v37 )
    {
LABEL_30:
      v38 = (unsigned __int8 *)v50;
      *(_QWORD *)(v24 + 48) = v50;
      if ( v38 )
        sub_B976B0((__int64)&v50, v38, v27);
      goto LABEL_12;
    }
LABEL_29:
    v46 = v27;
    sub_B91220(v27, v37);
    v27 = v46;
    goto LABEL_30;
  }
  v27 = v24 + 48;
  if ( (__int64 *)(v24 + 48) != &v50 )
  {
    v37 = *(_QWORD *)(v24 + 48);
    if ( v37 )
      goto LABEL_29;
  }
LABEL_12:
  *(_BYTE *)(a1 + 56) = 1;
  v28 = *(_QWORD *)(a1 + 48);
  v50 = 6;
  v51 = 0;
  v52 = (const char *)a2;
  if ( a2 != -4096 && a2 != -8192 )
    sub_BD73F0((__int64)&v50);
  v29 = *(unsigned int *)(v28 + 8);
  v30 = &v50;
  v31 = *(_QWORD *)v28;
  v32 = v29 + 1;
  v33 = *(_DWORD *)(v28 + 8);
  if ( v29 + 1 > (unsigned __int64)*(unsigned int *)(v28 + 12) )
  {
    v40 = v28 + 16;
    if ( v31 > (unsigned __int64)&v50 || (unsigned __int64)&v50 >= v31 + 24 * v29 )
    {
      v43 = (unsigned __int64 *)sub_C8D7D0(v28, v28 + 16, v29 + 1, 0x18u, &v49, v32);
      sub_F17F80(v28, v43);
      v44 = v49;
      if ( v40 == *(_QWORD *)v28 )
      {
        *(_QWORD *)v28 = v43;
      }
      else
      {
        v48 = v49;
        _libc_free(*(_QWORD *)v28);
        *(_QWORD *)v28 = v43;
        v44 = v48;
      }
      v29 = *(unsigned int *)(v28 + 8);
      *(_DWORD *)(v28 + 12) = v44;
      v31 = (unsigned __int64)v43;
      v30 = &v50;
      v33 = v29;
    }
    else
    {
      v41 = (char *)&v50 - v31;
      v47 = (unsigned __int64 *)sub_C8D7D0(v28, v28 + 16, v29 + 1, 0x18u, &v49, v32);
      sub_F17F80(v28, v47);
      v42 = v49;
      if ( v40 == *(_QWORD *)v28 )
      {
        *(_QWORD *)v28 = v47;
        *(_DWORD *)(v28 + 12) = v42;
      }
      else
      {
        v45 = v49;
        _libc_free(*(_QWORD *)v28);
        *(_QWORD *)v28 = v47;
        *(_DWORD *)(v28 + 12) = v45;
      }
      v31 = *(_QWORD *)v28;
      v29 = *(unsigned int *)(v28 + 8);
      v30 = (__int64 *)&v41[*(_QWORD *)v28];
      v33 = *(_DWORD *)(v28 + 8);
    }
  }
  v34 = (unsigned __int64 *)(v31 + 24 * v29);
  if ( v34 )
  {
    *v34 = 6;
    v35 = v30[2];
    v34[1] = 0;
    v34[2] = v35;
    if ( v35 != 0 && v35 != -4096 && v35 != -8192 )
      sub_BD6050(v34, *v30 & 0xFFFFFFFFFFFFFFF8LL);
    v33 = *(_DWORD *)(v28 + 8);
  }
  *(_DWORD *)(v28 + 8) = v33 + 1;
  if ( v52 != 0 && v52 + 4096 != 0 && v52 != (const char *)-8192LL )
    sub_BD60C0(&v50);
  return v18;
}
