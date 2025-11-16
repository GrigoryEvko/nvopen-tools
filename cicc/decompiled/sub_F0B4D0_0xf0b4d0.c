// Function: sub_F0B4D0
// Address: 0xf0b4d0
//
_QWORD *__fastcall sub_F0B4D0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r12
  __int64 v7; // rax
  __int64 v8; // rdi
  __int64 v9; // rsi
  unsigned int v10; // r10d
  _BYTE *v11; // r14
  char v12; // al
  __int64 v13; // rdi
  __int64 v14; // rdx
  _BYTE *v15; // rax
  __int64 *v16; // r11
  __int64 v17; // rdx
  _QWORD *v18; // r13
  __int64 v20; // r14
  __int64 v21; // rax
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 *v24; // r11
  unsigned int v25; // edx
  int v26; // eax
  __int64 v27; // rdx
  _BYTE *v28; // rax
  __int64 v29; // r15
  __int64 v30; // rax
  __int64 v31; // rax
  __int64 v32; // r15
  __int64 v33; // r10
  _BYTE *v34; // rax
  __int64 v35; // rdx
  int v36; // eax
  char v37; // cl
  int v38; // eax
  __int64 v39; // [rsp+8h] [rbp-B8h]
  char v40; // [rsp+17h] [rbp-A9h]
  __int64 *v41; // [rsp+18h] [rbp-A8h]
  unsigned int v42; // [rsp+20h] [rbp-A0h]
  unsigned int v43; // [rsp+20h] [rbp-A0h]
  unsigned int v44; // [rsp+20h] [rbp-A0h]
  unsigned int v45; // [rsp+20h] [rbp-A0h]
  _BYTE *v46; // [rsp+28h] [rbp-98h]
  __int64 v47; // [rsp+30h] [rbp-90h] BYREF
  _BYTE *v48; // [rsp+38h] [rbp-88h] BYREF
  __int64 v49; // [rsp+40h] [rbp-80h] BYREF
  unsigned int v50; // [rsp+48h] [rbp-78h]
  unsigned __int64 v51; // [rsp+50h] [rbp-70h] BYREF
  unsigned int v52; // [rsp+58h] [rbp-68h]
  unsigned __int64 v53; // [rsp+60h] [rbp-60h] BYREF
  __int64 v54; // [rsp+68h] [rbp-58h]
  __int16 v55; // [rsp+80h] [rbp-40h]

  if ( (*(_DWORD *)(a1 + 4) & 0x7FFFFFF) != 2 )
    return 0;
  v3 = *(_QWORD *)(a3 + 88);
  if ( !a2 )
    return 0;
  v7 = sub_BB5290(a2);
  if ( !sub_BCAC40(v7, 8) )
    return 0;
  v39 = *(_QWORD *)(a2 - 32LL * (*(_DWORD *)(a2 + 4) & 0x7FFFFFF));
  if ( !v39 )
    return 0;
  v8 = *(_QWORD *)(a2 + 32 * (1LL - (*(_DWORD *)(a2 + 4) & 0x7FFFFFF)));
  if ( *(_BYTE *)v8 == 17 )
  {
    v46 = (_BYTE *)(v8 + 24);
  }
  else
  {
    v17 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v8 + 8) + 8LL) - 17;
    if ( (unsigned int)v17 > 1 )
      return 0;
    if ( *(_BYTE *)v8 > 0x15u )
      return 0;
    v34 = sub_AD7630(v8, 0, v17);
    if ( !v34 || *v34 != 17 )
      return 0;
    v46 = v34 + 24;
  }
  v9 = *(_QWORD *)(a2 + 8);
  if ( (unsigned int)*(unsigned __int8 *)(v9 + 8) - 17 <= 1 )
    v9 = **(_QWORD **)(v9 + 16);
  v10 = sub_AE43F0(v3, v9);
  v11 = *(_BYTE **)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF)));
  v12 = *v11;
  if ( *v11 == 42 )
  {
    if ( !*((_QWORD *)v11 - 8) )
      return 0;
    v47 = *((_QWORD *)v11 - 8);
    v13 = *((_QWORD *)v11 - 4);
    if ( *(_BYTE *)v13 == 17 )
      goto LABEL_24;
    v27 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v13 + 8) + 8LL) - 17;
    if ( (unsigned int)v27 <= 1 && *(_BYTE *)v13 <= 0x15u )
    {
      v45 = v10;
      v28 = sub_AD7630(v13, 0, v27);
      v10 = v45;
      if ( v28 )
      {
        v16 = (__int64 *)(v28 + 24);
        if ( *v28 == 17 )
          goto LABEL_25;
      }
    }
    v12 = *v11;
  }
  if ( v12 != 58 || (v11[1] & 2) == 0 || !*((_QWORD *)v11 - 8) )
    return 0;
  v47 = *((_QWORD *)v11 - 8);
  v13 = *((_QWORD *)v11 - 4);
  if ( *(_BYTE *)v13 == 17 )
  {
LABEL_24:
    v16 = (__int64 *)(v13 + 24);
    goto LABEL_25;
  }
  v42 = v10;
  v14 = (unsigned int)*(unsigned __int8 *)(*(_QWORD *)(v13 + 8) + 8LL) - 17;
  if ( (unsigned int)v14 > 1 )
    return 0;
  if ( *(_BYTE *)v13 > 0x15u )
    return 0;
  v15 = sub_AD7630(v13, 0, v14);
  if ( !v15 || *v15 != 17 )
    return 0;
  v10 = v42;
  v16 = (__int64 *)(v15 + 24);
LABEL_25:
  if ( v10 != *((_DWORD *)v46 + 2) )
    return 0;
  v41 = v16;
  v43 = v10;
  if ( v10 != *((_DWORD *)v16 + 2) )
    return 0;
  v20 = *(_QWORD *)(a1 + 72);
  if ( *(_BYTE *)(v20 + 8) == 18 )
    return 0;
  v40 = sub_AE5020(v3, *(_QWORD *)(a1 + 72));
  v21 = sub_9208B0(v3, v20);
  v54 = v22;
  v53 = ((1LL << v40) + ((unsigned __int64)(v21 + 7) >> 3) - 1) >> v40 << v40;
  v23 = sub_CA1930(&v53);
  v24 = v41;
  v50 = v43;
  if ( v43 > 0x40 )
  {
    sub_C43690((__int64)&v49, v23, 0);
    v24 = v41;
  }
  else
  {
    v49 = v23;
  }
  sub_C472A0((__int64)&v53, (__int64)&v49, v24);
  sub_C45EE0((__int64)&v53, (__int64 *)v46);
  v25 = v54;
  v52 = v54;
  v51 = v53;
  if ( (unsigned int)v54 <= 0x40 )
  {
    if ( !v53 )
      goto LABEL_47;
    v18 = *(_QWORD **)(a2 + 16);
    if ( !v18 )
      goto LABEL_35;
    if ( v18[1] )
    {
      v18 = 0;
      goto LABEL_35;
    }
  }
  else
  {
    v44 = v54;
    v26 = sub_C444A0((__int64)&v51);
    v25 = v44;
    if ( v44 == v26 )
      goto LABEL_47;
    v18 = *(_QWORD **)(a2 + 16);
    if ( !v18 )
      goto LABEL_33;
    if ( v18[1] )
    {
      v18 = 0;
      goto LABEL_33;
    }
  }
  v18 = *(_QWORD **)(*(_QWORD *)(a1 + 32 * (1LL - (*(_DWORD *)(a1 + 4) & 0x7FFFFFF))) + 16LL);
  if ( !v18 )
    goto LABEL_51;
  if ( !v18[1] )
  {
LABEL_47:
    v29 = *(_QWORD *)(a3 + 32);
    v55 = 257;
    v48 = (_BYTE *)sub_ACCFD0(*(__int64 **)(v29 + 72), (__int64)&v51);
    v30 = sub_BCB2B0(*(_QWORD **)(v29 + 72));
    v31 = sub_921130((unsigned int **)v29, v30, v39, &v48, 1, (__int64)&v53, 0);
    v55 = 257;
    v32 = v31;
    v18 = sub_BD2C40(88, 2u);
    if ( !v18 )
    {
LABEL_50:
      v25 = v52;
      goto LABEL_51;
    }
    v33 = *(_QWORD *)(v32 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v33 + 8) - 17 <= 1 )
    {
LABEL_49:
      sub_B44260((__int64)v18, v33, 34, 2u, 0, 0);
      v18[9] = v20;
      v18[10] = sub_B4DC50(v20, (__int64)&v47, 1);
      sub_B4D9A0((__int64)v18, v32, &v47, 1, (__int64)&v53);
      goto LABEL_50;
    }
    v35 = *(_QWORD *)(v47 + 8);
    v36 = *(unsigned __int8 *)(v35 + 8);
    if ( v36 == 17 )
    {
      v37 = 0;
    }
    else
    {
      v37 = 1;
      if ( v36 != 18 )
        goto LABEL_49;
    }
    v38 = *(_DWORD *)(v35 + 32);
    BYTE4(v48) = v37;
    LODWORD(v48) = v38;
    v33 = sub_BCE1B0((__int64 *)v33, (__int64)v48);
    goto LABEL_49;
  }
  v18 = 0;
LABEL_51:
  if ( v25 > 0x40 )
  {
LABEL_33:
    if ( v51 )
      j_j___libc_free_0_0(v51);
  }
LABEL_35:
  if ( v50 > 0x40 && v49 )
    j_j___libc_free_0_0(v49);
  return v18;
}
