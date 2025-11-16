// Function: sub_8B2440
// Address: 0x8b2440
//
__int64 __fastcall sub_8B2440(unsigned __int64 a1, unsigned __int64 a2, char a3, unsigned int a4)
{
  char v5; // dl
  char v6; // al
  __int64 i; // r8
  _DWORD *j; // rsi
  __int64 **v9; // r13
  _QWORD *v10; // r12
  __int64 **v11; // rdi
  char v12; // bl
  __int64 v13; // rdx
  __int64 *v14; // r9
  __int64 v15; // r10
  __int64 *v16; // rcx
  char v17; // bl
  char v18; // al
  __int64 *v19; // r13
  __int64 *v20; // rax
  __int64 *v21; // rbx
  __int64 *v22; // rcx
  __int64 *v23; // rax
  __int64 v24; // rcx
  __int64 v25; // r8
  __int64 v26; // rax
  __int64 *v27; // rax
  __int64 *v28; // rax
  __int64 *v29; // rdi
  unsigned int v30; // r12d
  char v31; // dl
  _QWORD *v33; // rdx
  __int64 v34; // rax
  int v35; // eax
  __int64 **v36; // [rsp+10h] [rbp-B0h]
  __int64 v37; // [rsp+18h] [rbp-A8h]
  __int64 *v38; // [rsp+18h] [rbp-A8h]
  _QWORD *v40; // [rsp+30h] [rbp-90h]
  __int64 v41; // [rsp+38h] [rbp-88h]
  __int64 v42; // [rsp+40h] [rbp-80h]
  unsigned __int64 v43; // [rsp+48h] [rbp-78h]
  unsigned __int64 v44; // [rsp+50h] [rbp-70h]
  _BOOL4 v45; // [rsp+58h] [rbp-68h]
  __int64 v46; // [rsp+64h] [rbp-5Ch] BYREF
  int v47; // [rsp+6Ch] [rbp-54h] BYREF
  __int64 *v48; // [rsp+70h] [rbp-50h] BYREF
  __int64 *v49; // [rsp+78h] [rbp-48h] BYREF
  __int64 *v50; // [rsp+80h] [rbp-40h] BYREF
  __int64 *v51[7]; // [rsp+88h] [rbp-38h] BYREF

  v5 = *(_BYTE *)(a1 + 80);
  v43 = a1;
  v44 = a2;
  v46 = 0x100000001LL;
  v48 = 0;
  v49 = 0;
  v50 = 0;
  v51[0] = 0;
  if ( v5 == 16 )
  {
    v43 = **(_QWORD **)(a1 + 88);
    v5 = *(_BYTE *)(v43 + 80);
  }
  if ( v5 == 24 )
  {
    v43 = *(_QWORD *)(v43 + 88);
    v5 = *(_BYTE *)(v43 + 80);
  }
  v6 = *(_BYTE *)(a2 + 80);
  if ( v6 == 16 )
  {
    v44 = **(_QWORD **)(a2 + 88);
    v6 = *(_BYTE *)(v44 + 80);
  }
  if ( v6 == 24 )
  {
    v44 = *(_QWORD *)(v44 + 88);
    v6 = *(_BYTE *)(v44 + 80);
  }
  switch ( v5 )
  {
    case 4:
    case 5:
      v42 = *(_QWORD *)(*(_QWORD *)(v43 + 96) + 80LL);
      break;
    case 6:
      v42 = *(_QWORD *)(*(_QWORD *)(v43 + 96) + 32LL);
      break;
    case 9:
      v42 = *(_QWORD *)(*(_QWORD *)(v43 + 96) + 56LL);
      break;
    case 10:
      v42 = *(_QWORD *)(*(_QWORD *)(v43 + 96) + 56LL);
      break;
    case 19:
    case 20:
    case 21:
    case 22:
      v42 = *(_QWORD *)(v43 + 88);
      break;
    default:
      v42 = 0;
      break;
  }
  switch ( v6 )
  {
    case 4:
    case 5:
      v41 = *(_QWORD *)(*(_QWORD *)(v44 + 96) + 80LL);
      break;
    case 6:
      v41 = *(_QWORD *)(*(_QWORD *)(v44 + 96) + 32LL);
      break;
    case 9:
    case 10:
      v41 = *(_QWORD *)(*(_QWORD *)(v44 + 96) + 56LL);
      break;
    case 19:
    case 20:
    case 21:
    case 22:
      v41 = *(_QWORD *)(v44 + 88);
      break;
    default:
      BUG();
  }
  for ( i = *(_QWORD *)(*(_QWORD *)(v42 + 176) + 152LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  for ( j = *(_DWORD **)(*(_QWORD *)(v41 + 176) + 152LL); *((_BYTE *)j + 140) == 12; j = (_DWORD *)*((_QWORD *)j + 20) )
    ;
  v37 = i;
  v9 = (__int64 **)*((_QWORD *)j + 21);
  v36 = *(__int64 ***)(i + 168);
  v10 = **(_QWORD ***)(v41 + 328);
  v40 = **(_QWORD ***)(v42 + 328);
  v11 = (__int64 **)v43;
  v12 = sub_877F80(v43);
  sub_865900(v43);
  v45 = (a3 & 2) != 0;
  if ( v12 == 3 )
  {
    v11 = *(__int64 ***)(v37 + 160);
    j = (_DWORD *)*((_QWORD *)j + 20);
    sub_8B5380(
      (_DWORD)v11,
      (_DWORD)j,
      (unsigned int)&v50,
      (unsigned int)v51,
      (_DWORD)v10,
      (_DWORD)v40,
      0,
      0,
      a3 & 1,
      v45,
      (__int64)&v46,
      (__int64)&v46 + 4);
    goto LABEL_55;
  }
  v15 = (__int64)v36;
  if ( (a3 & 1) != 0 )
  {
    if ( !dword_4F077BC )
    {
      v11 = *(__int64 ***)(v37 + 160);
      j = (_DWORD *)*((_QWORD *)j + 20);
      sub_8B5380(
        (_DWORD)v11,
        (_DWORD)j,
        (unsigned int)&v50,
        (unsigned int)v51,
        (_DWORD)v10,
        (_DWORD)v40,
        0,
        0,
        1,
        v45,
        (__int64)&v46,
        (__int64)&v46 + 4);
    }
    v48 = *v36;
    v49 = *v9;
    goto LABEL_25;
  }
  v13 = (__int64)v9[5];
  j = &dword_4F077C4;
  v16 = v36[5];
  v48 = *v36;
  v17 = v13 != 0;
  v49 = *v9;
  v18 = 0;
  if ( dword_4F077C4 != 2
    || unk_4F07778 <= 201102 && (v18 = dword_4F07774, !dword_4F07774)
    || dword_4F077BC && (v18 = qword_4F077B4, !(_DWORD)qword_4F077B4) )
  {
LABEL_20:
    if ( (a3 & 0xC) != 0 )
    {
      v17 &= v18 ^ 1;
      if ( v16 )
      {
        j = (_DWORD *)v42;
        v11 = (__int64 **)v15;
        sub_88D870(v15, v42, &v48);
      }
      goto LABEL_23;
    }
    goto LABEL_25;
  }
  if ( (v16 != 0) == v17 )
  {
    v18 = 0;
    goto LABEL_20;
  }
  if ( !v16 || (*(_BYTE *)(v44 + 81) & 0x10) != 0 )
  {
    v18 = 0;
    if ( v13 )
    {
      v11 = (__int64 **)v43;
      if ( (*(_BYTE *)(v43 + 81) & 0x10) == 0 )
      {
        j = (_DWORD *)v41;
        v11 = v9;
        v38 = v16;
        sub_88D870((__int64)v9, v41, &v49);
        v15 = (__int64)v36;
        v18 = 1;
        v16 = v38;
      }
    }
    goto LABEL_20;
  }
  j = (_DWORD *)v42;
  v11 = v36;
  sub_88D870((__int64)v36, v42, &v48);
  if ( (a3 & 0xC) != 0 )
  {
LABEL_23:
    if ( v17 )
    {
      j = (_DWORD *)v41;
      v11 = v9;
      sub_88D870((__int64)v9, v41, &v49);
    }
  }
LABEL_25:
  v19 = 0;
  if ( (a3 & 4) != 0 )
  {
    v19 = v48;
    if ( v48 )
    {
      v13 = 0;
      while ( 1 )
      {
        v20 = (__int64 *)*v19;
        *v19 = v13;
        v13 = (__int64)v19;
        if ( !v20 )
          break;
        v19 = v20;
      }
    }
    v48 = v19;
  }
  v21 = 0;
  if ( (a3 & 8) != 0 )
  {
    v21 = v49;
    if ( v49 )
    {
      v22 = 0;
      while ( 1 )
      {
        v23 = (__int64 *)*v21;
        *v21 = (__int64)v22;
        v22 = v21;
        if ( !v23 )
          break;
        v21 = v23;
      }
    }
    v49 = v21;
  }
  v24 = (__int64)v48;
  v25 = (__int64)v49;
  if ( v48 )
  {
    v26 = (__int64)v49;
    do
    {
      if ( !v26 || (a3 & 1) == 0 && (*(_DWORD *)(v24 + 36) > a4 || *(_DWORD *)(v26 + 36) > a4) )
        break;
      j = *(_DWORD **)(v26 + 8);
      v11 = *(__int64 ***)(v24 + 8);
      sub_8B5380(
        (_DWORD)v11,
        (_DWORD)j,
        (unsigned int)&v50,
        (unsigned int)v51,
        (_DWORD)v10,
        (_DWORD)v40,
        *(_BYTE *)(v24 + 33) & 1,
        *(_BYTE *)(v26 + 33) & 1,
        a3 & 1,
        v45,
        (__int64)&v46,
        (__int64)&v46 + 4);
      if ( !v46 )
        break;
      v24 = *v48;
      v48 = (__int64 *)v24;
      v26 = *v49;
      v49 = (__int64 *)*v49;
    }
    while ( v24 );
  }
  if ( (a3 & 4) != 0 && v19 )
  {
    v24 = 0;
    while ( 1 )
    {
      v27 = (__int64 *)*v19;
      *v19 = v24;
      v24 = (__int64)v19;
      if ( !v27 )
        break;
      v19 = v27;
    }
  }
  if ( (a3 & 8) != 0 && v21 )
  {
    v24 = 0;
    while ( 1 )
    {
      v28 = (__int64 *)*v21;
      *v21 = v24;
      v24 = (__int64)v21;
      if ( !v28 )
        break;
      v21 = v28;
    }
  }
LABEL_55:
  if ( (_DWORD)v46 )
  {
    j = (_DWORD *)v44;
    v11 = &v50;
    LODWORD(v46) = sub_8B2240((__int64 *)&v50, v44, v10, 8u, a4) != 0;
  }
  if ( HIDWORD(v46) )
  {
    HIDWORD(v46) = 0;
    if ( (a3 & 2) != 0 || (v11 = v51, j = (_DWORD *)v43, sub_8B2240((__int64 *)v51, v43, v40, 8u, a4)) )
      HIDWORD(v46) = 1;
  }
  sub_864110((__int64)v11, (__int64)j, v13, v24, v25, v14);
  if ( !(_DWORD)v46 )
  {
    v29 = v50;
    v30 = -(HIDWORD(v46) != 0);
    goto LABEL_67;
  }
  v29 = v50;
  v30 = 1;
  if ( !HIDWORD(v46) )
    goto LABEL_67;
  if ( v48 )
  {
    if ( !v49 )
    {
      v30 = -1;
      if ( (*((_BYTE *)v48 + 33) & 1) != 0 )
        goto LABEL_67;
    }
  }
  else if ( v49 && (*((_BYTE *)v49 + 33) & 1) != 0 )
  {
    goto LABEL_67;
  }
  v31 = *(_BYTE *)(v41 + 160) & 0x10;
  if ( (*(_BYTE *)(v42 + 160) & 0x10) != 0 )
  {
    if ( v31 )
    {
      v29 = v50;
      v30 = sub_88D570(v50, v51[0]);
      goto LABEL_67;
    }
  }
  else
  {
    v30 = 1;
    if ( v31 )
      goto LABEL_67;
  }
  v30 = -((*(_BYTE *)(v42 + 160) & 0x10) != 0);
LABEL_67:
  if ( v29 )
    sub_725130(v29);
  if ( v51[0] )
    sub_725130(v51[0]);
  if ( dword_4D04494 )
  {
    if ( (a3 & 2) != 0 )
    {
      v33 = *(_QWORD **)(v41 + 104);
      v34 = v33[22];
      if ( v34 && (*(_QWORD *)(v34 + 16) || (*(_BYTE *)(*(_QWORD *)(*v33 + 88LL) + 160LL) & 0x20) != 0) && v30 == -1 )
      {
        v35 = sub_6F3270(v43, v44, &v47);
        return (unsigned int)-(v47 != 0 || v35 == -1);
      }
    }
    else if ( !v30 )
    {
      v30 = v46;
      if ( (_DWORD)v46 )
      {
        v30 = HIDWORD(v46);
        if ( HIDWORD(v46) )
          return (unsigned int)sub_6F3270(v43, v44, &v47);
      }
    }
  }
  return v30;
}
