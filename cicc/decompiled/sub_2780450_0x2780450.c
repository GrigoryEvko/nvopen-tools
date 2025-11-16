// Function: sub_2780450
// Address: 0x2780450
//
__int64 __fastcall sub_2780450(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  __int64 v6; // rbx
  __int64 *v7; // rax
  __int64 v8; // r13
  _QWORD *v9; // rdx
  unsigned int v10; // r14d
  unsigned int v11; // eax
  __int64 v12; // rsi
  __int64 v13; // r15
  __int64 v14; // rdx
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rcx
  __int64 v18; // r8
  __int64 v19; // rdi
  _BYTE *v20; // rdi
  __int64 v21; // r15
  _BYTE *v22; // r9
  __int64 v23; // r15
  unsigned __int64 *v24; // rbx
  __int64 *v25; // rdx
  __int64 v26; // rcx
  __int64 v27; // r8
  __int64 v28; // r9
  __int64 *v29; // rax
  __int64 v30; // rax
  unsigned __int64 v31; // rdx
  __int64 v33; // r8
  char v34; // dl
  __int64 *v35; // rax
  _BYTE *v36; // rdi
  __int64 v37; // [rsp+18h] [rbp-F8h]
  __int64 v38; // [rsp+18h] [rbp-F8h]
  __int64 v40; // [rsp+38h] [rbp-D8h]
  int v41; // [rsp+44h] [rbp-CCh]
  unsigned __int8 *v43; // [rsp+58h] [rbp-B8h] BYREF
  _BYTE *v44; // [rsp+60h] [rbp-B0h] BYREF
  __int64 v45; // [rsp+68h] [rbp-A8h]
  _QWORD *v46; // [rsp+70h] [rbp-A0h] BYREF
  __int64 v47; // [rsp+78h] [rbp-98h]
  _QWORD v48[4]; // [rsp+80h] [rbp-90h] BYREF
  __int64 v49; // [rsp+A0h] [rbp-70h] BYREF
  __int64 *v50; // [rsp+A8h] [rbp-68h]
  __int64 v51; // [rsp+B0h] [rbp-60h]
  int v52; // [rsp+B8h] [rbp-58h]
  char v53; // [rsp+BCh] [rbp-54h]
  char v54; // [rsp+C0h] [rbp-50h] BYREF

  v6 = a1;
  if ( a4 == *(_QWORD *)(a3 - 32) )
  {
    v35 = (__int64 *)sub_AA48A0(a4);
    v8 = sub_ACD6D0(v35);
  }
  else
  {
    v7 = (__int64 *)sub_AA48A0(a4);
    v8 = sub_ACD720(v7);
  }
  v9 = v48;
  v41 = (a4 != *(_QWORD *)(a3 - 32)) + 28;
  v50 = (__int64 *)&v54;
  v46 = v48;
  v49 = 0;
  v51 = 4;
  v52 = 0;
  v53 = 1;
  v47 = 0x400000001LL;
  v48[0] = a2;
  v10 = 0;
  v40 = a1 + 120;
  v11 = 1;
  while ( 1 )
  {
    v12 = *(_QWORD *)(v6 + 256);
    v13 = v9[v11 - 1];
    LODWORD(v47) = v11 - 1;
    v44 = (_BYTE *)v8;
    v43 = (unsigned __int8 *)v13;
    sub_27801B0(v40, v12, &v43, &v44);
    v14 = *(_QWORD *)(v6 + 16);
    v44 = (_BYTE *)a5;
    v45 = a4;
    if ( (unsigned int)sub_F57230(v13, v8, v14, (__int64 *)&v44) )
      v10 = 1;
    if ( v41 != 28 )
    {
      if ( !v13 )
        goto LABEL_14;
      v15 = *(_QWORD *)(v13 + 8);
      if ( (unsigned int)*(unsigned __int8 *)(v15 + 8) - 17 <= 1 )
        v15 = **(_QWORD **)(v15 + 16);
      if ( !sub_BCAC40(v15, 1) )
        goto LABEL_14;
      if ( *(_BYTE *)v13 == 58 )
        break;
      if ( *(_BYTE *)v13 != 86 )
        goto LABEL_14;
      v37 = *(_QWORD *)(v13 - 96);
      if ( *(_QWORD *)(v37 + 8) != *(_QWORD *)(v13 + 8) )
        goto LABEL_14;
      v36 = *(_BYTE **)(v13 - 64);
      if ( *v36 > 0x15u )
        goto LABEL_14;
      v21 = *(_QWORD *)(v13 - 32);
      if ( !sub_AD7A80(v36, 1, v16, v17, v18) )
        goto LABEL_14;
      goto LABEL_25;
    }
    if ( !v13 )
      goto LABEL_14;
    v19 = *(_QWORD *)(v13 + 8);
    if ( (unsigned int)*(unsigned __int8 *)(v19 + 8) - 17 <= 1 )
      v19 = **(_QWORD **)(v19 + 16);
    if ( !sub_BCAC40(v19, 1) )
      goto LABEL_14;
    if ( *(_BYTE *)v13 == 57 )
      break;
    if ( *(_BYTE *)v13 != 86 )
      goto LABEL_14;
    v37 = *(_QWORD *)(v13 - 96);
    if ( *(_QWORD *)(v37 + 8) != *(_QWORD *)(v13 + 8) )
      goto LABEL_14;
    v20 = *(_BYTE **)(v13 - 32);
    if ( *v20 > 0x15u )
      goto LABEL_14;
    v21 = *(_QWORD *)(v13 - 64);
    if ( !sub_AC30F0((__int64)v20) )
      goto LABEL_14;
LABEL_25:
    v22 = (_BYTE *)v37;
    if ( v21 )
      goto LABEL_26;
LABEL_14:
    v11 = v47;
    if ( !(_DWORD)v47 )
      goto LABEL_41;
LABEL_15:
    v9 = v46;
  }
  if ( (*(_BYTE *)(v13 + 7) & 0x40) != 0 )
    v33 = *(_QWORD *)(v13 - 8);
  else
    v33 = v13 - 32LL * (*(_DWORD *)(v13 + 4) & 0x7FFFFFF);
  v22 = *(_BYTE **)v33;
  if ( !*(_QWORD *)v33 )
    goto LABEL_14;
  v21 = *(_QWORD *)(v33 + 32);
  if ( !v21 )
    goto LABEL_14;
LABEL_26:
  v45 = v21;
  v23 = (__int64)v22;
  v44 = v22;
  v38 = v6;
  v24 = (unsigned __int64 *)&v44;
  if ( *v22 > 0x1Cu && sub_2778480((__int64)v22) )
    goto LABEL_31;
LABEL_28:
  while ( ++v24 != (unsigned __int64 *)&v46 )
  {
    while ( 1 )
    {
      v23 = *v24;
      if ( *(_BYTE *)*v24 <= 0x1Cu || !sub_2778480(*v24) )
        break;
LABEL_31:
      if ( !v53 )
        goto LABEL_51;
      v29 = v50;
      v26 = HIDWORD(v51);
      v25 = &v50[HIDWORD(v51)];
      if ( v50 != v25 )
      {
        while ( v23 != *v29 )
        {
          if ( v25 == ++v29 )
            goto LABEL_35;
        }
        goto LABEL_28;
      }
LABEL_35:
      if ( HIDWORD(v51) < (unsigned int)v51 )
      {
        ++HIDWORD(v51);
        *v25 = v23;
        ++v49;
      }
      else
      {
LABEL_51:
        sub_C8CC70((__int64)&v49, v23, (__int64)v25, v26, v27, v28);
        if ( !v34 )
          goto LABEL_28;
      }
      v30 = (unsigned int)v47;
      v31 = (unsigned int)v47 + 1LL;
      if ( v31 > HIDWORD(v47) )
      {
        sub_C8D5F0((__int64)&v46, v48, v31, 8u, v27, v28);
        v30 = (unsigned int)v47;
      }
      ++v24;
      v46[v30] = v23;
      LODWORD(v47) = v47 + 1;
      if ( v24 == (unsigned __int64 *)&v46 )
        goto LABEL_40;
    }
  }
LABEL_40:
  v11 = v47;
  v6 = v38;
  if ( (_DWORD)v47 )
    goto LABEL_15;
LABEL_41:
  if ( !v53 )
    _libc_free((unsigned __int64)v50);
  if ( v46 != v48 )
    _libc_free((unsigned __int64)v46);
  return v10;
}
