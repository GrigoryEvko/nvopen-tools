// Function: sub_27DC180
// Address: 0x27dc180
//
__int64 __fastcall sub_27DC180(__int64 **a1, _QWORD *a2, unsigned __int8 *a3, unsigned int a4)
{
  _QWORD *v4; // rbx
  _QWORD *v5; // rdx
  int v6; // eax
  unsigned int v7; // r13d
  unsigned __int64 v9; // rax
  int v10; // edx
  unsigned __int8 *v11; // rax
  __int64 v12; // r9
  unsigned __int8 *v13; // rax
  unsigned __int8 *v14; // r15
  _QWORD *v15; // rcx
  _BYTE *v16; // r10
  size_t v17; // r8
  _QWORD *v18; // rax
  __int64 v19; // rax
  unsigned __int8 *v20; // r10
  unsigned __int8 *v21; // rax
  __int64 v22; // rax
  __int64 v23; // r11
  __int64 v24; // r8
  _QWORD *v25; // rcx
  int v26; // edx
  unsigned __int8 **v27; // r9
  unsigned __int64 v28; // rax
  __int64 v29; // rax
  int v30; // edx
  unsigned int v31; // edx
  __int64 v32; // rax
  __int64 v33; // rax
  _QWORD *v34; // rdi
  unsigned int v35; // eax
  bool v36; // cf
  __int64 v37; // [rsp+8h] [rbp-B8h]
  size_t n; // [rsp+10h] [rbp-B0h]
  size_t na; // [rsp+10h] [rbp-B0h]
  int src; // [rsp+18h] [rbp-A8h]
  _BYTE *srca; // [rsp+18h] [rbp-A8h]
  unsigned __int8 *srcb; // [rsp+18h] [rbp-A8h]
  __int64 v43; // [rsp+20h] [rbp-A0h]
  _QWORD *v44; // [rsp+20h] [rbp-A0h]
  _QWORD *v45; // [rsp+20h] [rbp-A0h]
  __int64 v46; // [rsp+20h] [rbp-A0h]
  unsigned int v48; // [rsp+30h] [rbp-90h]
  unsigned int v49; // [rsp+34h] [rbp-8Ch]
  size_t v51; // [rsp+48h] [rbp-78h] BYREF
  _QWORD *v52; // [rsp+50h] [rbp-70h] BYREF
  __int64 v53; // [rsp+58h] [rbp-68h]
  _QWORD v54[2]; // [rsp+60h] [rbp-60h] BYREF
  __int64 v55; // [rsp+70h] [rbp-50h]
  __int64 v56; // [rsp+78h] [rbp-48h]
  __int64 v57; // [rsp+80h] [rbp-40h]

  v4 = (_QWORD *)a2[7];
  v5 = a2 + 6;
  v49 = a4;
  if ( v4 != a2 + 6 )
  {
    v6 = 0;
    while ( v4 )
    {
      if ( *((_BYTE *)v4 - 24) != 84 )
        goto LABEL_10;
      if ( ++v6 > (unsigned int)qword_4FFDA68 )
        return (unsigned int)-1;
      v4 = (_QWORD *)v4[1];
      if ( v4 == v5 )
        goto LABEL_9;
    }
LABEL_79:
    BUG();
  }
LABEL_9:
  v4 = 0;
LABEL_10:
  v9 = a2[6] & 0xFFFFFFFFFFFFFFF8LL;
  if ( v5 == (_QWORD *)v9 )
  {
    v11 = 0;
  }
  else
  {
    if ( !v9 )
      goto LABEL_79;
    v10 = *(unsigned __int8 *)(v9 - 24);
    v11 = (unsigned __int8 *)(v9 - 24);
    if ( (unsigned int)(v10 - 30) >= 0xB )
      v11 = 0;
  }
  v48 = 0;
  if ( a3 == v11 )
  {
    if ( *a3 == 32 )
    {
      v49 = a4 + 6;
      v48 = 6;
    }
    else if ( *a3 == 33 )
    {
      v49 = a4 + 8;
      v48 = 8;
    }
  }
  v7 = 0;
  while ( 1 )
  {
    v12 = (__int64)(v4 - 3);
    v13 = 0;
    if ( v4 )
      v13 = (unsigned __int8 *)(v4 - 3);
    v14 = v13;
    if ( v13 == a3 )
      break;
    v15 = *(_QWORD **)(a2[9] + 40LL);
    v52 = v54;
    v16 = (_BYTE *)v15[29];
    v17 = v15[30];
    if ( &v16[v17] && !v16 )
      sub_426248((__int64)"basic_string::_M_construct null not valid");
    v51 = v15[30];
    if ( v17 > 0xF )
    {
      n = v17;
      srca = v16;
      v44 = v15;
      v33 = sub_22409D0((__int64)&v52, &v51, 0);
      v15 = v44;
      v16 = srca;
      v52 = (_QWORD *)v33;
      v34 = (_QWORD *)v33;
      v17 = n;
      v54[0] = v51;
LABEL_57:
      v45 = v15;
      memcpy(v34, v16, v17);
      v17 = v51;
      v18 = v52;
      v15 = v45;
      goto LABEL_24;
    }
    if ( v17 == 1 )
    {
      LOBYTE(v54[0]) = *v16;
      v18 = v54;
    }
    else
    {
      if ( v17 )
      {
        v34 = v54;
        goto LABEL_57;
      }
      v18 = v54;
    }
LABEL_24:
    v53 = v17;
    *((_BYTE *)v18 + v17) = 0;
    v55 = v15[33];
    v56 = v15[34];
    v57 = v15[35];
    if ( (unsigned int)(v55 - 42) <= 1 && (unsigned int)*v14 - 30 <= 0xA )
    {
      if ( v52 != v54 )
        j_j___libc_free_0((unsigned __int64)v52);
      return (unsigned int)-1;
    }
    if ( v52 != v54 )
      j_j___libc_free_0((unsigned __int64)v52);
    if ( v7 > v49 )
      return v7;
    if ( *(_BYTE *)(*((_QWORD *)v14 + 1) + 8LL) == 11 && (unsigned __int8)sub_B463C0((__int64)v14, (__int64)a2)
      || *v14 == 85
      && ((unsigned __int8)sub_A73ED0((_QWORD *)v14 + 9, 27)
       || (unsigned __int8)sub_B49560((__int64)v14, 27)
       || (unsigned __int8)sub_A73ED0((_QWORD *)v14 + 9, 6)
       || (unsigned __int8)sub_B49560((__int64)v14, 6)) )
    {
      return (unsigned int)-1;
    }
    v19 = 32LL * (*((_DWORD *)v14 + 1) & 0x7FFFFFF);
    if ( (v14[7] & 0x40) != 0 )
    {
      v20 = (unsigned __int8 *)*((_QWORD *)v14 - 1);
      v21 = &v20[v19];
    }
    else
    {
      v20 = &v14[-v19];
      v21 = v14;
    }
    v22 = v21 - v20;
    v52 = v54;
    v53 = 0x400000000LL;
    v23 = v22 >> 5;
    v24 = v22 >> 5;
    if ( (unsigned __int64)v22 > 0x80 )
    {
      v37 = v22 >> 5;
      na = v22;
      srcb = v20;
      v46 = v22 >> 5;
      sub_C8D5F0((__int64)&v52, v54, v22 >> 5, 8u, v24, v12);
      v27 = (unsigned __int8 **)v52;
      v26 = v53;
      LODWORD(v23) = v46;
      v20 = srcb;
      v22 = na;
      v24 = v37;
      v25 = &v52[(unsigned int)v53];
    }
    else
    {
      v25 = v54;
      v26 = 0;
      v27 = (unsigned __int8 **)v54;
    }
    if ( v22 > 0 )
    {
      v28 = 0;
      do
      {
        v25[v28 / 8] = *(_QWORD *)&v20[4 * v28];
        v28 += 8LL;
        --v24;
      }
      while ( v24 );
      v27 = (unsigned __int8 **)v52;
      v26 = v53;
    }
    LODWORD(v53) = v23 + v26;
    v29 = sub_DFCEF0(a1, v14, v27, (unsigned int)(v23 + v26), 3);
    if ( v52 != v54 )
    {
      src = v30;
      v43 = v29;
      _libc_free((unsigned __int64)v52);
      v30 = src;
      v29 = v43;
    }
    if ( v30 || v29 )
    {
      v31 = v7 + 1;
      if ( *v14 == 85 )
      {
        v32 = *((_QWORD *)v14 - 4);
        if ( v32
          && !*(_BYTE *)v32
          && *(_QWORD *)(v32 + 24) == *((_QWORD *)v14 + 10)
          && (*(_BYTE *)(v32 + 33) & 0x20) != 0 )
        {
          v7 += 2;
          if ( (unsigned int)*(unsigned __int8 *)(*((_QWORD *)v14 + 1) + 8LL) - 17 <= 1 )
            v7 = v31;
        }
        else
        {
          v7 += 4;
        }
      }
      else
      {
        ++v7;
      }
    }
    v4 = (_QWORD *)v4[1];
  }
  v35 = v7 - v48;
  v36 = v48 < v7;
  v7 = 0;
  if ( v36 )
    return v35;
  return v7;
}
