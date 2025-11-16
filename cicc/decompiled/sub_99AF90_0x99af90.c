// Function: sub_99AF90
// Address: 0x99af90
//
char __fastcall sub_99AF90(_BYTE **a1, __int64 a2)
{
  _BYTE **v2; // r12
  __int64 v3; // r13
  __int64 v4; // rax
  int v5; // ebx
  _BYTE *v6; // r14
  unsigned __int64 v7; // rax
  int v8; // edx
  unsigned __int64 v9; // rax
  int v10; // r14d
  int v11; // edx
  unsigned __int64 v12; // rax
  int v13; // edx
  __int64 v14; // r14
  unsigned __int64 v15; // rax
  int v16; // r10d
  int v17; // edx
  char result; // al
  __int64 v19; // r14
  __int64 v20; // rax
  bool v21; // zf
  char v22; // al
  _BYTE *v23; // rdi
  __int64 v24; // rax
  char v25; // al
  _BYTE *v26; // rdi
  __int64 v27; // rax
  char v28; // al
  _BYTE *v29; // rdi
  __int64 v30; // rax
  char v31; // al
  unsigned __int64 v32; // rax
  int v33; // edx
  unsigned __int64 v34; // rax
  int v35; // edx
  unsigned __int64 v36; // rax
  int v37; // edx
  _BYTE **v38; // [rsp+8h] [rbp-78h]
  _BYTE **v39; // [rsp+10h] [rbp-70h]
  char v40; // [rsp+1Fh] [rbp-61h]
  char v41; // [rsp+1Fh] [rbp-61h]
  char v42; // [rsp+1Fh] [rbp-61h]
  char v43; // [rsp+1Fh] [rbp-61h]
  _BYTE *v44; // [rsp+20h] [rbp-60h]
  _BYTE *v45; // [rsp+20h] [rbp-60h]
  int v46; // [rsp+28h] [rbp-58h]
  _BYTE **v47; // [rsp+28h] [rbp-58h]
  __int64 v48; // [rsp+30h] [rbp-50h] BYREF
  __int64 v49; // [rsp+38h] [rbp-48h] BYREF
  unsigned __int64 v50; // [rsp+44h] [rbp-3Ch]
  int v51; // [rsp+4Ch] [rbp-34h]

  v2 = a1;
  v38 = &a1[a2];
  v3 = (8 * a2) >> 5;
  v4 = (8 * a2) >> 3;
  if ( v3 > 0 )
  {
    v40 = 1;
    v5 = 0;
    v39 = &a1[4 * v3];
    while ( 1 )
    {
      v14 = (__int64)*v2;
      v15 = sub_99AEC0(*v2, &v48, &v49, 0, 0);
      v50 = v15;
      v16 = v15;
      v51 = v17;
      if ( !(_DWORD)v15 || (_DWORD)v15 == 7 || (_DWORD)v15 == 8 || v5 != (_DWORD)v15 && v5 != 0 )
        break;
      if ( *(_BYTE *)v14 == 86
        && ((*(_BYTE *)(v14 + 7) & 0x40) != 0
          ? (v19 = *(_QWORD *)(v14 - 8))
          : (v19 = v14 - 32LL * (*(_DWORD *)(v14 + 4) & 0x7FFFFFF)),
            (v20 = *(_QWORD *)(*(_QWORD *)v19 + 16LL)) != 0) )
      {
        v21 = *(_QWORD *)(v20 + 8) == 0;
        v22 = v40;
        if ( !v21 )
          v22 = 0;
        v41 = v22;
      }
      else
      {
        v41 = 0;
      }
      v6 = v2[1];
      v46 = v16;
      v7 = sub_99AEC0(v6, &v48, &v49, 0, 0);
      v50 = v7;
      v5 = v7;
      v51 = v8;
      if ( !(_DWORD)v7 || (_DWORD)v7 == 7 || (_DWORD)v7 == 8 || (_DWORD)v7 != v46 )
      {
        ++v2;
        v5 = v46;
        break;
      }
      if ( *v6 == 86
        && ((v6[7] & 0x40) != 0
          ? (v23 = (_BYTE *)*((_QWORD *)v6 - 1))
          : (v23 = &v6[-32 * (*((_DWORD *)v6 + 1) & 0x7FFFFFF)]),
            (v24 = *(_QWORD *)(*(_QWORD *)v23 + 16LL)) != 0) )
      {
        v21 = *(_QWORD *)(v24 + 8) == 0;
        v25 = v41;
        if ( !v21 )
          v25 = 0;
        v42 = v25;
      }
      else
      {
        v42 = 0;
      }
      v47 = v2 + 2;
      v44 = v2[2];
      v9 = sub_99AEC0(v44, &v48, &v49, 0, 0);
      v50 = v9;
      v10 = v9;
      v51 = v11;
      if ( !(_DWORD)v9 || (_DWORD)v9 == 7 || (_DWORD)v9 == 8 || v5 != (_DWORD)v9 )
        goto LABEL_30;
      if ( *v44 == 86
        && ((v44[7] & 0x40) != 0
          ? (v26 = (_BYTE *)*((_QWORD *)v44 - 1))
          : (v26 = &v44[-32 * (*((_DWORD *)v44 + 1) & 0x7FFFFFF)]),
            (v27 = *(_QWORD *)(*(_QWORD *)v26 + 16LL)) != 0) )
      {
        v21 = *(_QWORD *)(v27 + 8) == 0;
        v28 = v42;
        if ( !v21 )
          v28 = 0;
        v43 = v28;
      }
      else
      {
        v43 = 0;
      }
      v47 = v2 + 3;
      v45 = v2[3];
      v12 = sub_99AEC0(v45, &v48, &v49, 0, 0);
      v50 = v12;
      v5 = v12;
      v51 = v13;
      if ( !(_DWORD)v12 || (_DWORD)v12 == 7 || (_DWORD)v12 == 8 || (_DWORD)v12 != v10 )
      {
        v5 = v10;
LABEL_30:
        result = 0;
        if ( v38 != v47 )
          return result;
        goto LABEL_31;
      }
      if ( *v45 == 86
        && ((v45[7] & 0x40) != 0
          ? (v29 = (_BYTE *)*((_QWORD *)v45 - 1))
          : (v29 = &v45[-32 * (*((_DWORD *)v45 + 1) & 0x7FFFFFF)]),
            (v30 = *(_QWORD *)(*(_QWORD *)v29 + 16LL)) != 0) )
      {
        v21 = *(_QWORD *)(v30 + 8) == 0;
        v31 = v43;
        if ( !v21 )
          v31 = 0;
        v2 += 4;
        v40 = v31;
        if ( v2 == v39 )
        {
LABEL_57:
          v4 = v38 - v2;
          goto LABEL_58;
        }
      }
      else
      {
        v40 = 0;
        v2 += 4;
        if ( v2 == v39 )
          goto LABEL_57;
      }
    }
LABEL_27:
    result = 0;
    if ( v38 != v2 )
      return result;
    goto LABEL_31;
  }
  v5 = 0;
LABEL_58:
  if ( v4 != 2 )
  {
    if ( v4 != 3 )
    {
      if ( v4 != 1 )
        goto LABEL_31;
      goto LABEL_61;
    }
    v34 = sub_99AEC0(*v2, &v48, &v49, 0, 0);
    v50 = v34;
    v51 = v35;
    if ( !(_DWORD)v34 || (_DWORD)v34 == 7 || (_DWORD)v34 == 8 || v5 != (_DWORD)v34 && v5 != 0 )
      goto LABEL_27;
    ++v2;
    v5 = v34;
  }
  v36 = sub_99AEC0(*v2, &v48, &v49, 0, 0);
  v50 = v36;
  v51 = v37;
  if ( !(_DWORD)v36 || (_DWORD)v36 == 7 || (_DWORD)v36 == 8 || v5 != 0 && v5 != (_DWORD)v36 )
    goto LABEL_27;
  ++v2;
  v5 = v36;
LABEL_61:
  v32 = sub_99AEC0(*v2, &v48, &v49, 0, 0);
  v50 = v32;
  v51 = v33;
  if ( !(_DWORD)v32 || (_DWORD)v32 == 7 || (_DWORD)v32 == 8 || v5 != 0 && (_DWORD)v32 != v5 )
    goto LABEL_27;
  v5 = v32;
LABEL_31:
  switch ( v5 )
  {
    case 1:
      result = 74;
      break;
    case 2:
      result = 110;
      break;
    case 3:
      result = 73;
      break;
    case 4:
      result = 109;
      break;
    case 5:
      result = -8;
      break;
    case 6:
      result = -19;
      break;
    default:
      BUG();
  }
  return result;
}
