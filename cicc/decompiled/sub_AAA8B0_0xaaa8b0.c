// Function: sub_AAA8B0
// Address: 0xaaa8b0
//
__int64 __fastcall sub_AAA8B0(__int64 a1, __int64 a2, _DWORD *a3, __int64 a4)
{
  _DWORD *v4; // r14
  __int64 v7; // r13
  unsigned int *v8; // rdi
  char v9; // r8
  __int64 v10; // r15
  __int64 v11; // rdx
  __int64 v12; // r9
  _DWORD *v13; // rax
  __int64 v14; // rsi
  unsigned int *v15; // rax
  unsigned int v16; // edi
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rax
  unsigned int v20; // r13d
  __int64 v21; // rax
  __int64 v22; // rax
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rax
  char v26; // al
  _BYTE *v27; // rdi
  __int64 v28; // rsi
  __int64 v29; // r12
  __int64 v31; // rax
  __int64 v32; // rax
  __int64 v33; // [rsp+0h] [rbp-180h]
  int v34; // [rsp+18h] [rbp-168h]
  char v35; // [rsp+24h] [rbp-15Ch]
  __int64 v36; // [rsp+28h] [rbp-158h]
  __int64 v37; // [rsp+28h] [rbp-158h]
  size_t v38; // [rsp+38h] [rbp-148h]
  _BYTE *v39; // [rsp+40h] [rbp-140h] BYREF
  __int64 v40; // [rsp+48h] [rbp-138h]
  _BYTE v41[304]; // [rsp+50h] [rbp-130h] BYREF

  v4 = a3;
  v7 = *(_QWORD *)(a1 + 8);
  v8 = &a3[a4];
  LODWORD(v38) = a4;
  v9 = *(_BYTE *)(v7 + 8);
  v10 = *(_QWORD *)(v7 + 24);
  BYTE4(v38) = v9 == 18;
  v11 = (4 * a4) >> 4;
  v12 = (4 * a4) >> 2;
  if ( v11 > 0 )
  {
    v13 = v4;
    v14 = (__int64)&v4[4 * v11];
    while ( *v13 == -1 )
    {
      if ( v13[1] != -1 )
      {
        ++v13;
        break;
      }
      if ( v13[2] != -1 )
      {
        v13 += 2;
        break;
      }
      if ( v13[3] != -1 )
      {
        v13 += 3;
        break;
      }
      v13 += 4;
      if ( v13 == (_DWORD *)v14 )
      {
        v14 = v8 - v13;
        goto LABEL_40;
      }
    }
    if ( v8 != v13 )
    {
      v15 = v4;
      goto LABEL_14;
    }
    goto LABEL_44;
  }
  v14 = (4 * a4) >> 2;
  v13 = v4;
LABEL_40:
  if ( v14 != 2 )
  {
    if ( v14 != 3 )
    {
      if ( v14 != 1 )
      {
LABEL_44:
        v31 = sub_BCE1B0(v10, v38);
        return sub_ACADE0(v31);
      }
      goto LABEL_43;
    }
    if ( *v13 != -1 )
      goto LABEL_53;
    ++v13;
  }
  if ( *v13 != -1 )
    goto LABEL_53;
  ++v13;
LABEL_43:
  if ( *v13 == -1 )
    goto LABEL_44;
LABEL_53:
  if ( v8 == v13 )
    goto LABEL_44;
  v15 = v4;
  if ( v11 > 0 )
  {
LABEL_14:
    while ( !*v15 )
    {
      v14 = v15[1];
      if ( (_DWORD)v14 )
      {
        if ( v8 != v15 + 1 )
          goto LABEL_16;
        goto LABEL_31;
      }
      if ( v15[2] )
      {
        if ( v8 != v15 + 2 )
          goto LABEL_16;
        goto LABEL_31;
      }
      if ( v15[3] )
      {
        if ( v8 != v15 + 3 )
          goto LABEL_16;
        goto LABEL_31;
      }
      v15 += 4;
      if ( !--v11 )
      {
        v12 = v8 - v15;
        goto LABEL_28;
      }
    }
    goto LABEL_15;
  }
LABEL_28:
  switch ( v12 )
  {
    case 2LL:
      goto LABEL_58;
    case 3LL:
      if ( *v15 )
        goto LABEL_15;
      ++v15;
LABEL_58:
      v14 = *v15;
      if ( (_DWORD)v14 )
        goto LABEL_15;
      ++v15;
      goto LABEL_60;
    case 1LL:
LABEL_60:
      v11 = *v15;
      if ( !(_DWORD)v11 )
        break;
LABEL_15:
      if ( v8 != v15 )
        goto LABEL_16;
      break;
  }
LABEL_31:
  v34 = a4;
  v35 = *(_BYTE *)(v7 + 8);
  v23 = sub_BD5C60(a1, v14, v11);
  v24 = sub_BCCE00(v23, 32);
  v25 = sub_AD64C0(v24, 0, 0);
  v37 = sub_AD5840(a1, v25, 0);
  v26 = sub_AC30F0(v37);
  v14 = v37;
  LODWORD(a4) = v34;
  if ( v26 )
  {
    v32 = sub_BCE1B0(v10, v38);
    return sub_AC9350(v32);
  }
  if ( v35 != 18 )
    return sub_AD5E10(v38);
  v9 = *(_BYTE *)(v7 + 8);
LABEL_16:
  if ( v9 == 18 )
    return 0;
  v16 = *(_DWORD *)(v7 + 32);
  v39 = v41;
  v40 = 0x2000000000LL;
  if ( (_DWORD)a4 )
  {
    v36 = (__int64)&v4[(unsigned int)(a4 - 1) + 1];
    do
    {
      v20 = *v4;
      if ( *v4 == -1 || 2 * v16 <= v20 )
      {
        v19 = sub_ACA8A0(v10);
      }
      else if ( v20 >= v16 )
      {
        v17 = sub_BD5C60(a2, v14, v11);
        v18 = sub_BCCE00(v17, 32);
        v14 = sub_AD64C0(v18, v20 - v16, 0);
        v19 = sub_AD5840(a2, v14, 0);
      }
      else
      {
        v21 = sub_BD5C60(a1, v14, v11);
        v22 = sub_BCCE00(v21, 32);
        v14 = sub_AD64C0(v22, (int)v20, 0);
        v19 = sub_AD5840(a1, v14, 0);
      }
      v11 = (unsigned int)v40;
      if ( (unsigned __int64)(unsigned int)v40 + 1 > HIDWORD(v40) )
      {
        v14 = (__int64)v41;
        v33 = v19;
        sub_C8D5F0(&v39, v41, (unsigned int)v40 + 1LL, 8);
        v11 = (unsigned int)v40;
        v19 = v33;
      }
      ++v4;
      *(_QWORD *)&v39[8 * v11] = v19;
      LODWORD(v40) = v40 + 1;
    }
    while ( (_DWORD *)v36 != v4 );
    v27 = v39;
    v28 = (unsigned int)v40;
  }
  else
  {
    v27 = v41;
    v28 = 0;
  }
  v29 = sub_AD3730(v27, v28);
  if ( v39 != v41 )
    _libc_free(v39, v28);
  return v29;
}
