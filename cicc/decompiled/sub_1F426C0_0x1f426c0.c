// Function: sub_1F426C0
// Address: 0x1f426c0
//
__int64 __fastcall sub_1F426C0(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, unsigned int *a6, _BYTE *a7)
{
  unsigned int v8; // r12d
  __int64 v9; // rdx
  unsigned int v10; // ebx
  unsigned __int8 v11; // al
  __int64 v12; // rsi
  unsigned __int8 v13; // al
  __int64 v14; // rcx
  __int64 v15; // r9
  __int64 v16; // rdx
  __int64 v17; // rbx
  __int64 v18; // r8
  char v19; // r13
  char v20; // r14
  unsigned int v21; // eax
  __int64 v22; // rcx
  __int64 v23; // r9
  __int64 v24; // r8
  unsigned __int64 v25; // rdx
  unsigned int v27; // eax
  __int64 v28; // rcx
  __int64 v29; // r8
  __int64 v30; // r9
  __int64 v31; // rdx
  unsigned int v32; // ebx
  unsigned int v33; // eax
  int v34; // ebx
  int v35; // ebx
  unsigned int v36; // eax
  char v37; // al
  __int64 v38; // rdx
  __int64 v39; // [rsp-10h] [rbp-F0h]
  __int64 v40; // [rsp-8h] [rbp-E8h]
  __int64 v43; // [rsp+18h] [rbp-C8h]
  unsigned __int8 v44; // [rsp+28h] [rbp-B8h]
  __int64 v45; // [rsp+28h] [rbp-B8h]
  unsigned int v46; // [rsp+28h] [rbp-B8h]
  unsigned int v47; // [rsp+28h] [rbp-B8h]
  __int64 v48; // [rsp+28h] [rbp-B8h]
  unsigned int v49; // [rsp+28h] [rbp-B8h]
  __int64 v50; // [rsp+30h] [rbp-B0h] BYREF
  __int64 v51; // [rsp+38h] [rbp-A8h]
  char v52; // [rsp+4Bh] [rbp-95h] BYREF
  _BYTE v53[4]; // [rsp+4Ch] [rbp-94h] BYREF
  __int64 v54; // [rsp+50h] [rbp-90h] BYREF
  __int64 v55; // [rsp+58h] [rbp-88h]
  __int64 v56; // [rsp+60h] [rbp-80h] BYREF
  __int64 v57; // [rsp+68h] [rbp-78h]
  __int64 v58; // [rsp+70h] [rbp-70h] BYREF
  __int64 v59; // [rsp+78h] [rbp-68h]
  __int64 v60; // [rsp+80h] [rbp-60h] BYREF
  __int64 v61; // [rsp+88h] [rbp-58h]
  __int64 v62; // [rsp+90h] [rbp-50h] BYREF
  __int64 v63; // [rsp+98h] [rbp-48h]
  __int64 v64; // [rsp+A0h] [rbp-40h]

  v50 = a3;
  v51 = a4;
  if ( (_BYTE)a3 )
    v8 = word_42F2F80[(unsigned __int8)(a3 - 14)];
  else
    v8 = sub_1F58D30(&v50);
  sub_1F40D10((__int64)&v62, a1, a2, v50, v51);
  if ( v8 != 1 && ((_BYTE)v62 == 7 || (_BYTE)v62 == 1) )
  {
    sub_1F40D10((__int64)&v62, a1, a2, v50, v51);
    v37 = v63;
    v38 = v64;
    if ( (_BYTE)v63 )
    {
      if ( *(_QWORD *)(a1 + 8LL * (unsigned __int8)v63 + 120) )
      {
        v8 = 1;
        *(_BYTE *)a5 = v63;
        *(_QWORD *)(a5 + 8) = v38;
        *a7 = v37;
        *a6 = 1;
        return v8;
      }
    }
  }
  if ( (_BYTE)v50 )
  {
    switch ( (char)v50 )
    {
      case 14:
      case 15:
      case 16:
      case 17:
      case 18:
      case 19:
      case 20:
      case 21:
      case 22:
      case 23:
      case 56:
      case 57:
      case 58:
      case 59:
      case 60:
      case 61:
        v44 = 2;
        break;
      case 24:
      case 25:
      case 26:
      case 27:
      case 28:
      case 29:
      case 30:
      case 31:
      case 32:
      case 62:
      case 63:
      case 64:
      case 65:
      case 66:
      case 67:
        v44 = 3;
        break;
      case 33:
      case 34:
      case 35:
      case 36:
      case 37:
      case 38:
      case 39:
      case 40:
      case 68:
      case 69:
      case 70:
      case 71:
      case 72:
      case 73:
        v44 = 4;
        break;
      case 41:
      case 42:
      case 43:
      case 44:
      case 45:
      case 46:
      case 47:
      case 48:
      case 74:
      case 75:
      case 76:
      case 77:
      case 78:
      case 79:
        v44 = 5;
        break;
      case 49:
      case 50:
      case 51:
      case 52:
      case 53:
      case 54:
      case 80:
      case 81:
      case 82:
      case 83:
      case 84:
      case 85:
        v44 = 6;
        break;
      case 55:
        v44 = 7;
        break;
      case 86:
      case 87:
      case 88:
      case 98:
      case 99:
      case 100:
        v44 = 8;
        break;
      case 89:
      case 90:
      case 91:
      case 92:
      case 93:
      case 101:
      case 102:
      case 103:
      case 104:
      case 105:
        v44 = 9;
        break;
      case 94:
      case 95:
      case 96:
      case 97:
      case 106:
      case 107:
      case 108:
      case 109:
        v44 = 10;
        break;
    }
    v43 = 0;
  }
  else
  {
    v44 = sub_1F596B0(&v50);
    v43 = v9;
  }
  v10 = 1;
  if ( v8 && (v8 & (v8 - 1)) == 0 && v8 != 1 )
  {
    v10 = v8;
    v8 = 1;
    do
    {
      v11 = sub_1D15020(v44, v10);
      if ( v11 || (v11 = sub_1F593D0(a2, v44, v43, v10)) != 0 )
      {
        if ( *(_QWORD *)(a1 + 8LL * v11 + 120) )
          break;
      }
      v10 >>= 1;
      v8 *= 2;
    }
    while ( v10 != 1 );
  }
  v12 = v10;
  *a6 = v8;
  v13 = sub_1D15020(v44, v10);
  v16 = v44;
  if ( v13 )
  {
    LOBYTE(v54) = v13;
    v55 = 0;
    goto LABEL_16;
  }
  v12 = v44;
  v13 = sub_1F593D0(a2, v44, v43, v10);
  LOBYTE(v54) = v13;
  v55 = v16;
  if ( v13 )
  {
LABEL_16:
    if ( *(_QWORD *)(a1 + 8LL * v13 + 120) )
      goto LABEL_17;
  }
  LOBYTE(v54) = v44;
  v55 = v43;
LABEL_17:
  v17 = v54;
  v18 = v55;
  *(_QWORD *)a5 = v54;
  *(_QWORD *)(a5 + 8) = v18;
  v56 = v17;
  v57 = v18;
  if ( (_BYTE)v17 )
  {
    v19 = *(_BYTE *)(a1 + (unsigned __int8)v17 + 1155);
  }
  else
  {
    v48 = v18;
    if ( (unsigned __int8)sub_1F58D20(&v56) )
    {
      v12 = a2;
      LOBYTE(v62) = 0;
      v63 = 0;
      LOBYTE(v58) = 0;
      sub_1F426C0(a1, a2, v56, v57, (unsigned int)&v62, (unsigned int)&v60, (__int64)&v58);
      v19 = v58;
      v17 = v54;
      v18 = v55;
    }
    else
    {
      v12 = a1;
      sub_1F40D10((__int64)&v62, a1, a2, v17, v48);
      v34 = v64;
      LOBYTE(v58) = v63;
      v59 = v64;
      if ( (_BYTE)v63 )
      {
        v19 = *(_BYTE *)(a1 + (unsigned __int8)v63 + 1155);
        v17 = v54;
        v18 = v55;
      }
      else if ( (unsigned __int8)sub_1F58D20(&v58) )
      {
        LOBYTE(v62) = 0;
        v63 = 0;
        v53[0] = 0;
        sub_1F426C0(a1, a2, v58, v34, (unsigned int)&v62, (unsigned int)&v60, (__int64)v53);
        v14 = v39;
        v12 = v40;
        v19 = v53[0];
        v17 = v54;
        v18 = v55;
      }
      else
      {
        v12 = a1;
        sub_1F40D10((__int64)&v62, a1, a2, v58, v59);
        v35 = v64;
        LOBYTE(v60) = v63;
        v61 = v64;
        if ( (_BYTE)v63 )
        {
          v19 = *(_BYTE *)(a1 + (unsigned __int8)v63 + 1155);
        }
        else if ( (unsigned __int8)sub_1F58D20(&v60) )
        {
          v12 = a2;
          LOBYTE(v62) = 0;
          v63 = 0;
          v52 = 0;
          sub_1F426C0(a1, a2, v60, v35, (unsigned int)&v62, (unsigned int)v53, (__int64)&v52);
          v19 = v52;
          v16 = v40;
        }
        else
        {
          sub_1F40D10((__int64)&v62, a1, a2, v60, v61);
          v12 = a2;
          v19 = sub_1D5E9F0(a1, a2, (unsigned __int8)v63, v64);
        }
        v17 = v54;
        v18 = v55;
      }
    }
  }
  v20 = v54;
  LOBYTE(v58) = v19;
  v45 = v18;
  *a7 = v19;
  if ( v20 )
    v21 = sub_1F3E310(&v54);
  else
    v21 = sub_1F58D40(&v54, v12, v16, v14, v18, v15);
  v24 = v45;
  v25 = v21;
  if ( !v21 || (v21 & (v21 - 1)) != 0 )
    v25 = (((((((((v21 | ((unsigned __int64)v21 >> 1)) >> 2) | v21 | ((unsigned __int64)v21 >> 1)) >> 4)
             | ((v21 | ((unsigned __int64)v21 >> 1)) >> 2)
             | v21
             | ((unsigned __int64)v21 >> 1)) >> 8)
           | ((((v21 | ((unsigned __int64)v21 >> 1)) >> 2) | v21 | ((unsigned __int64)v21 >> 1)) >> 4)
           | ((v21 | ((unsigned __int64)v21 >> 1)) >> 2)
           | v21
           | ((unsigned __int64)v21 >> 1)) >> 16)
         | ((((((v21 | ((unsigned __int64)v21 >> 1)) >> 2) | v21 | ((unsigned __int64)v21 >> 1)) >> 4)
           | ((v21 | ((unsigned __int64)v21 >> 1)) >> 2)
           | v21
           | ((unsigned __int64)v21 >> 1)) >> 8)
         | ((((v21 | ((unsigned __int64)v21 >> 1)) >> 2) | v21 | ((unsigned __int64)v21 >> 1)) >> 4)
         | ((v21 | ((unsigned __int64)v21 >> 1)) >> 2)
         | v21
         | (v21 >> 1))
        + 1;
  LOBYTE(v60) = v19;
  v61 = 0;
  v62 = v17;
  v63 = v45;
  if ( v19 == v20 )
  {
    if ( v20 || !v45 )
      return v8;
    goto LABEL_49;
  }
  if ( !v19 )
  {
LABEL_49:
    v49 = v25;
    v36 = sub_1F58D40(&v60, v12, v25, v22, v24, v23);
    v31 = v49;
    v32 = v36;
    goto LABEL_36;
  }
  v46 = v25;
  v27 = sub_1F3E310(&v60);
  v31 = v46;
  v32 = v27;
LABEL_36:
  v47 = v31;
  if ( v20 )
    v33 = sub_1F3E310(&v62);
  else
    v33 = sub_1F58D40(&v62, v12, v31, v28, v29, v30);
  if ( v33 > v32 )
    v8 *= v47 / (unsigned int)sub_1F3E310(&v58);
  return v8;
}
