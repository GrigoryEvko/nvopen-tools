// Function: sub_8C3A50
// Address: 0x8c3a50
//
void __fastcall sub_8C3A50(__int64 a1)
{
  __int64 i2; // rbx
  _QWORD *v2; // r13
  __int64 **v3; // r12
  __int64 *v4; // r14
  char v5; // al
  __int64 *v6; // rax
  __int64 *i; // rdx
  __int64 *v8; // r14
  char v9; // al
  __int64 *v10; // rax
  __int64 *j; // rdx
  __int64 *v12; // r14
  char v13; // al
  __int64 *v14; // rax
  __int64 *k; // rdx
  __int64 *v16; // r14
  char v17; // al
  __int64 *v18; // rax
  __int64 *m; // rdx
  __int64 *v20; // r14
  char v21; // al
  __int64 *v22; // rax
  __int64 *n; // rdx
  __int64 *v24; // r14
  char v25; // al
  __int64 *v26; // rax
  __int64 *ii; // rdx
  __int64 *v28; // r14
  char v29; // al
  __int64 *v30; // rax
  __int64 *jj; // rdx
  __int64 *v32; // r14
  char v33; // al
  __int64 *v34; // rax
  __int64 *kk; // rdx
  __int64 *v36; // r14
  char v37; // al
  __int64 *v38; // rax
  __int64 *mm; // rdx
  __int64 *v40; // r14
  char v41; // al
  __int64 *v42; // rax
  __int64 *nn; // rdx
  __int64 *v44; // r14
  char v45; // al
  __int64 *v46; // rax
  __int64 *i1; // rdx

  if ( *(_BYTE *)(a1 + 28) == 3 && (*(_BYTE *)(a1 - 8) & 8) == 0 )
  {
    v2 = (_QWORD *)a1;
    v3 = (__int64 **)sub_85EB10(a1);
    if ( (*(_BYTE *)(a1 - 8) & 3) == 3 )
    {
      sub_8C3650((__int64 *)a1, 0x17u, 0);
      v2 = *(_QWORD **)(a1 - 24);
      if ( (*(_BYTE *)(v2 - 1) & 2) != 0 )
        v2 = (_QWORD *)*(v2 - 3);
    }
    v4 = v3[3];
    if ( v4 )
    {
      v5 = *((_BYTE *)v4 - 8);
      if ( (v5 & 8) != 0 )
      {
        v6 = (__int64 *)v2[12];
        for ( i = 0; v6; v6 = (__int64 *)v6[15] )
          i = v6;
        v3[3] = i;
      }
      else
      {
        if ( (v5 & 3) == 3 )
        {
          sub_8C3650(v3[3], 2u, 0);
          v4 = (__int64 *)*(v4 - 3);
          if ( (*(_BYTE *)(v4 - 1) & 2) != 0 )
            v4 = (__int64 *)*(v4 - 3);
        }
        v3[3] = v4;
      }
    }
    v8 = v3[4];
    if ( v8 )
    {
      v9 = *((_BYTE *)v8 - 8);
      if ( (v9 & 8) != 0 )
      {
        v10 = (__int64 *)v2[13];
        for ( j = 0; v10; v10 = (__int64 *)v10[14] )
          j = v10;
        v3[4] = j;
      }
      else
      {
        if ( (v9 & 3) == 3 )
        {
          sub_8C3650(v3[4], 6u, 0);
          v8 = (__int64 *)*(v8 - 3);
          if ( (*(_BYTE *)(v8 - 1) & 2) != 0 )
            v8 = (__int64 *)*(v8 - 3);
        }
        v3[4] = v8;
      }
    }
    v12 = v3[5];
    if ( v12 )
    {
      v13 = *((_BYTE *)v12 - 8);
      if ( (v13 & 8) != 0 )
      {
        v14 = (__int64 *)v2[14];
        for ( k = 0; v14; v14 = (__int64 *)v14[14] )
          k = v14;
        v3[5] = k;
      }
      else
      {
        if ( (v13 & 3) == 3 )
        {
          sub_8C3650(v3[5], 7u, 0);
          v12 = (__int64 *)*(v12 - 3);
          if ( (*(_BYTE *)(v12 - 1) & 2) != 0 )
            v12 = (__int64 *)*(v12 - 3);
        }
        v3[5] = v12;
      }
    }
    v16 = v3[6];
    if ( v16 )
    {
      v17 = *((_BYTE *)v16 - 8);
      if ( (v17 & 8) != 0 )
      {
        v18 = (__int64 *)v2[18];
        for ( m = 0; v18; v18 = (__int64 *)v18[14] )
          m = v18;
        v3[6] = m;
      }
      else
      {
        if ( (v17 & 3) == 3 )
        {
          sub_8C3650(v3[6], 0xBu, 0);
          v16 = (__int64 *)*(v16 - 3);
          if ( (*(_BYTE *)(v16 - 1) & 2) != 0 )
            v16 = (__int64 *)*(v16 - 3);
        }
        v3[6] = v16;
      }
    }
    v20 = v3[7];
    if ( v20 )
    {
      v21 = *((_BYTE *)v20 - 8);
      if ( (v21 & 8) != 0 )
      {
        v22 = (__int64 *)v2[19];
        for ( n = 0; v22; v22 = (__int64 *)v22[14] )
          n = v22;
        v3[7] = n;
      }
      else
      {
        if ( (v21 & 3) == 3 )
        {
          sub_8C3650(v3[7], 0x2Bu, 0);
          v20 = (__int64 *)*(v20 - 3);
          if ( (*(_BYTE *)(v20 - 1) & 2) != 0 )
            v20 = (__int64 *)*(v20 - 3);
        }
        v3[7] = v20;
      }
    }
    v24 = v3[8];
    if ( v24 )
    {
      v25 = *((_BYTE *)v24 - 8);
      if ( (v25 & 8) != 0 )
      {
        v26 = (__int64 *)v2[24];
        for ( ii = 0; v26; v26 = (__int64 *)*v26 )
          ii = v26;
        v3[8] = ii;
      }
      else
      {
        if ( (v25 & 3) == 3 )
        {
          sub_8C3650(v3[8], 0x1Eu, 0);
          v24 = (__int64 *)*(v24 - 3);
          if ( (*(_BYTE *)(v24 - 1) & 2) != 0 )
            v24 = (__int64 *)*(v24 - 3);
        }
        v3[8] = v24;
      }
    }
    v28 = v3[9];
    if ( v28 )
    {
      v29 = *((_BYTE *)v28 - 8);
      if ( (v29 & 8) != 0 )
      {
        v30 = (__int64 *)v2[21];
        for ( jj = 0; v30; v30 = (__int64 *)v30[14] )
          jj = v30;
        v3[9] = jj;
      }
      else
      {
        if ( (v29 & 3) == 3 )
        {
          sub_8C3650(v3[9], 0x1Cu, 0);
          v28 = (__int64 *)*(v28 - 3);
          if ( (*(_BYTE *)(v28 - 1) & 2) != 0 )
            v28 = (__int64 *)*(v28 - 3);
        }
        v3[9] = v28;
      }
    }
    v32 = v3[11];
    if ( v32 )
    {
      v33 = *((_BYTE *)v32 - 8);
      if ( (v33 & 8) != 0 )
      {
        v34 = (__int64 *)v2[23];
        for ( kk = 0; v34; v34 = (__int64 *)*v34 )
          kk = v34;
        v3[11] = kk;
      }
      else
      {
        if ( (v33 & 3) == 3 )
        {
          sub_8C3650(v3[11], 0x1Du, 0);
          v32 = (__int64 *)*(v32 - 3);
          if ( (*(_BYTE *)(v32 - 1) & 2) != 0 )
            v32 = (__int64 *)*(v32 - 3);
        }
        v3[11] = v32;
      }
    }
    v36 = v3[10];
    if ( v36 )
    {
      v37 = *((_BYTE *)v36 - 8);
      if ( (v37 & 8) != 0 )
      {
        v38 = (__int64 *)v2[22];
        for ( mm = 0; v38; v38 = (__int64 *)*v38 )
          mm = v38;
        v3[10] = mm;
      }
      else
      {
        if ( (v37 & 3) == 3 )
        {
          sub_8C3650(v3[10], 0x1Du, 0);
          v36 = (__int64 *)*(v36 - 3);
          if ( (*(_BYTE *)(v36 - 1) & 2) != 0 )
            v36 = (__int64 *)*(v36 - 3);
        }
        v3[10] = v36;
      }
    }
    v40 = v3[12];
    if ( v40 )
    {
      v41 = *((_BYTE *)v40 - 8);
      if ( (v41 & 8) != 0 )
      {
        v42 = (__int64 *)v2[29];
        for ( nn = 0; v42; v42 = (__int64 *)*v42 )
          nn = v42;
        v3[12] = nn;
      }
      else
      {
        if ( (v41 & 3) == 3 )
        {
          sub_8C3650(v3[12], 0x3Au, 0);
          v40 = (__int64 *)*(v40 - 3);
          if ( (*(_BYTE *)(v40 - 1) & 2) != 0 )
            v40 = (__int64 *)*(v40 - 3);
        }
        v3[12] = v40;
      }
    }
    v44 = v3[13];
    if ( v44 )
    {
      v45 = *((_BYTE *)v44 - 8);
      if ( (v45 & 8) != 0 )
      {
        v46 = (__int64 *)v2[34];
        for ( i1 = 0; v46; v46 = (__int64 *)v46[14] )
          i1 = v46;
        v3[13] = i1;
      }
      else
      {
        if ( (v45 & 3) == 3 )
        {
          sub_8C3650(v3[13], 0x3Bu, 0);
          v44 = (__int64 *)*(v44 - 3);
          if ( (*(_BYTE *)(v44 - 1) & 2) != 0 )
            v44 = (__int64 *)*(v44 - 3);
        }
        v3[13] = v44;
      }
    }
  }
  for ( i2 = *(_QWORD *)(a1 + 168); i2; i2 = *(_QWORD *)(i2 + 112) )
  {
    if ( (*(_BYTE *)(i2 + 124) & 1) == 0 )
      sub_8C3A50(*(_QWORD *)(i2 + 128));
  }
}
