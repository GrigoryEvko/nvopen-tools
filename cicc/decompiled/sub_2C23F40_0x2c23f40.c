// Function: sub_2C23F40
// Address: 0x2c23f40
//
__int64 __fastcall sub_2C23F40(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v4; // r14
  __int64 v5; // r13
  __int64 v6; // rax
  __int64 v7; // r12
  __int64 *v8; // rax
  __int64 result; // rax
  char v10; // al
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // rax
  bool v14; // al
  __int64 v15; // rax
  __int64 v16; // r8
  __int64 v17; // rdx
  __int64 v18; // r9
  _QWORD *v19; // rax
  _QWORD *v20; // rsi
  __int64 v21; // rcx
  __int64 v22; // rdx
  _QWORD *v23; // rdx
  bool v24; // al
  __int64 v25; // r9
  bool v26; // r15
  __int64 v27; // rax
  __int64 v28; // rax
  __int64 *v29; // r10
  __int64 *v30; // rax
  __int64 v31; // [rsp+8h] [rbp-E8h]
  __int64 v32; // [rsp+20h] [rbp-D0h]
  bool v33; // [rsp+28h] [rbp-C8h]
  bool v34; // [rsp+28h] [rbp-C8h]
  __int64 v35; // [rsp+28h] [rbp-C8h]
  bool v36; // [rsp+30h] [rbp-C0h]
  __int64 v37; // [rsp+30h] [rbp-C0h]
  __int64 v38; // [rsp+40h] [rbp-B0h]
  __int64 v39; // [rsp+50h] [rbp-A0h] BYREF
  __int64 v40; // [rsp+58h] [rbp-98h] BYREF
  __int64 v41; // [rsp+60h] [rbp-90h] BYREF
  int v42; // [rsp+68h] [rbp-88h]
  __int64 *v43; // [rsp+70h] [rbp-80h] BYREF
  unsigned __int64 v44; // [rsp+78h] [rbp-78h]
  unsigned __int64 v45; // [rsp+80h] [rbp-70h] BYREF
  __int64 *v46; // [rsp+88h] [rbp-68h]
  __int64 *v47; // [rsp+90h] [rbp-60h] BYREF
  unsigned __int64 v48; // [rsp+98h] [rbp-58h]
  __int64 *v49; // [rsp+A0h] [rbp-50h] BYREF
  __int64 *v50; // [rsp+A8h] [rbp-48h]
  _QWORD *v51; // [rsp+B0h] [rbp-40h]
  __int64 *v52; // [rsp+B8h] [rbp-38h]

  v4 = a3 + 16;
  v5 = *(_QWORD *)(a1 + 136);
  v36 = sub_2BFB0D0(**(_QWORD **)(a1 + 48));
  v38 = sub_2BFD6A0(v4, a1 + 96);
  v6 = sub_2BFD6A0(v4, a1 + 96);
  v7 = sub_2AAEDF0(v6, a2);
  if ( v36 )
  {
    sub_2BFD6A0(v4, **(_QWORD **)(a1 + 48));
    return sub_DFD2D0(*(__int64 **)a3, 57, v7);
  }
  if ( (unsigned int)sub_BCB060(v38) != 1 )
  {
LABEL_3:
    v8 = (__int64 *)sub_2BFD6A0(v4, **(_QWORD **)(a1 + 48));
    sub_BCE1B0(v8, a2);
    return sub_DFD2D0(*(__int64 **)a3, 57, v7);
  }
  v42 = 64;
  v41 = 0;
  sub_9865C0((__int64)&v47, (__int64)&v41);
  v49 = &v40;
  v50 = &v39;
  sub_9865C0((__int64)&v43, (__int64)&v47);
  v45 = (unsigned __int64)v49;
  v46 = v50;
  sub_969240((__int64 *)&v47);
  v47 = &v40;
  v48 = (unsigned __int64)&v39;
  sub_9865C0((__int64)&v49, (__int64)&v43);
  v51 = (_QWORD *)v45;
  v52 = v46;
  sub_969240((__int64 *)&v43);
  sub_969240(&v41);
  v10 = *(_BYTE *)(a1 + 8);
  if ( v10 == 4 )
  {
    v11 = *(_BYTE *)(a1 + 160);
    if ( v11 != 83 )
      goto LABEL_9;
    v27 = **(_QWORD **)(a1 + 48);
    if ( !v27 )
      goto LABEL_14;
    *(_QWORD *)v48 = v27;
    v28 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL);
    if ( v28 )
    {
      *v47 = v28;
      goto LABEL_37;
    }
    v10 = *(_BYTE *)(a1 + 8);
    if ( v10 != 9 )
    {
      if ( v10 == 4 )
      {
        v11 = *(_BYTE *)(a1 + 160);
LABEL_9:
        if ( v11 != 57 )
          goto LABEL_14;
        goto LABEL_10;
      }
      goto LABEL_60;
    }
  }
  else if ( v10 != 9 )
  {
LABEL_60:
    if ( v10 != 24 )
      goto LABEL_14;
    goto LABEL_10;
  }
  if ( **(_BYTE **)(a1 + 136) != 86 )
    goto LABEL_14;
LABEL_10:
  v12 = **(_QWORD **)(a1 + 48);
  if ( v12 )
  {
    *v52 = v12;
    v13 = *(_QWORD *)(*(_QWORD *)(a1 + 48) + 8LL);
    if ( v13 )
    {
      *v51 = v13;
      sub_9865C0((__int64)&v43, (__int64)&v49);
      if ( !sub_2C23EC0((__int64)&v43, *(_QWORD *)(*(_QWORD *)(a1 + 48) + 16LL)) )
      {
        sub_969240((__int64 *)&v43);
        goto LABEL_14;
      }
      sub_969240((__int64 *)&v43);
LABEL_37:
      if ( (unsigned int)v50 <= 0x40 )
        goto LABEL_21;
      v14 = 1;
      goto LABEL_18;
    }
  }
LABEL_14:
  LODWORD(v45) = 64;
  v44 = 1;
  v43 = &v40;
  v46 = &v39;
  v14 = sub_2C1BCC0((__int64)&v43, a1);
  if ( (unsigned int)v45 > 0x40 && v44 )
  {
    v33 = v14;
    j_j___libc_free_0_0(v44);
    v14 = v33;
  }
  if ( (unsigned int)v50 > 0x40 )
  {
LABEL_18:
    if ( v49 )
    {
      v34 = v14;
      j_j___libc_free_0_0((unsigned __int64)v49);
      v14 = v34;
    }
  }
  if ( !v14 )
    goto LABEL_3;
LABEL_21:
  v35 = sub_2BF1540(a3, v39);
  v15 = sub_2BF1540(a3, v40);
  v17 = *(unsigned int *)(a1 + 56);
  v43 = (__int64 *)&v45;
  v18 = v15;
  v17 *= 8;
  v44 = 0x200000000LL;
  v19 = *(_QWORD **)(a1 + 48);
  v20 = (_QWORD *)((char *)v19 + v17);
  v21 = v17 >> 3;
  v22 = v17 >> 5;
  if ( !v22 )
  {
LABEL_40:
    if ( v21 != 2 )
    {
      if ( v21 != 3 )
      {
        if ( v21 != 1 )
          goto LABEL_44;
        goto LABEL_43;
      }
      if ( !*(_QWORD *)(*v19 + 40LL) )
        goto LABEL_28;
      ++v19;
    }
    if ( !*(_QWORD *)(*v19 + 40LL) )
      goto LABEL_28;
    ++v19;
LABEL_43:
    if ( !*(_QWORD *)(*v19 + 40LL) )
      goto LABEL_28;
LABEL_44:
    v31 = v18;
    sub_C8D5F0((__int64)&v43, &v45, 3u, 8u, v16, v18);
    v29 = (__int64 *)(v5 - 96);
    v18 = v31;
    v30 = &v43[(unsigned int)v44];
    do
    {
      if ( v30 )
        *v30 = *v29;
      v29 += 4;
      ++v30;
    }
    while ( (__int64 *)v5 != v29 );
    LODWORD(v44) = v44 + 3;
    goto LABEL_29;
  }
  v23 = &v19[4 * v22];
  while ( *(_QWORD *)(*v19 + 40LL) )
  {
    if ( !*(_QWORD *)(v19[1] + 40LL) )
    {
      ++v19;
      break;
    }
    if ( !*(_QWORD *)(v19[2] + 40LL) )
    {
      v19 += 2;
      break;
    }
    if ( !*(_QWORD *)(v19[3] + 40LL) )
    {
      v19 += 3;
      break;
    }
    v19 += 4;
    if ( v23 == v19 )
    {
      v21 = v20 - v19;
      goto LABEL_40;
    }
  }
LABEL_28:
  if ( v20 == v19 )
    goto LABEL_44;
LABEL_29:
  v32 = v18;
  LODWORD(v49) = 64;
  v47 = &v40;
  v48 = 1;
  v50 = &v39;
  v24 = sub_2C1BCC0((__int64)&v47, a1);
  v25 = v32;
  v26 = v24;
  if ( (unsigned int)v49 > 0x40 && v48 )
  {
    j_j___libc_free_0_0(v48);
    v25 = v32;
  }
  result = sub_DFD800(
             *(_QWORD *)a3,
             28 - ((unsigned int)!v26 - 1),
             v7,
             *(_DWORD *)(a3 + 176),
             v35,
             v25,
             v43,
             (unsigned int)v44,
             v5,
             0);
  if ( v43 != (__int64 *)&v45 )
  {
    v37 = result;
    _libc_free((unsigned __int64)v43);
    return v37;
  }
  return result;
}
