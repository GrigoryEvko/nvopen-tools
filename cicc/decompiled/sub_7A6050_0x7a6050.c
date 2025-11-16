// Function: sub_7A6050
// Address: 0x7a6050
//
__int64 __fastcall sub_7A6050(__int64 a1, __int64 a2, __int64 a3, char **a4, __int64 a5, __int64 a6)
{
  char *v9; // r14
  char v10; // si
  __int64 v11; // rdi
  __int64 v12; // r13
  __int64 v13; // rdi
  char v14; // al
  unsigned int v15; // r13d
  __int64 v16; // rsi
  char v18; // al
  __int64 v19; // rdx
  char v20; // al
  __int64 v21; // rdx
  __int64 v22; // rax
  __int64 v23; // rsi
  const __m128i **v24; // rdx
  _QWORD *v25; // r13
  __int64 i; // rax
  __int64 v27; // rcx
  __int64 v28; // rsi
  __int64 j; // rax
  __int64 v30; // rcx
  __int64 v31; // rsi
  __int64 k; // rax
  __int64 v33; // rcx
  __int64 v34; // rcx
  __int64 m; // r13
  __int64 v36; // rax
  __int64 n; // rsi
  unsigned int v38; // eax
  int v39; // eax
  __int64 v40; // [rsp+0h] [rbp-70h]
  __int64 v41; // [rsp+0h] [rbp-70h]
  __int64 v42; // [rsp+0h] [rbp-70h]
  __int64 v43; // [rsp+8h] [rbp-68h]
  const __m128i **v44; // [rsp+10h] [rbp-60h]
  const __m128i **v45; // [rsp+10h] [rbp-60h]
  const __m128i **v46; // [rsp+10h] [rbp-60h]
  const __m128i **v47; // [rsp+10h] [rbp-60h]
  __int64 v48; // [rsp+10h] [rbp-60h]
  __int64 v50; // [rsp+20h] [rbp-50h] BYREF
  __int64 v51; // [rsp+28h] [rbp-48h]
  __int64 v52; // [rsp+30h] [rbp-40h]

  v9 = *a4;
  v10 = **a4;
  v11 = *((_QWORD *)*a4 + 1);
  if ( v10 == 48 )
  {
    v18 = *(_BYTE *)(v11 + 8);
    if ( v18 == 1 )
    {
      v11 = *(_QWORD *)(v11 + 32);
      v10 = 2;
    }
    else if ( v18 == 2 )
    {
      v11 = *(_QWORD *)(v11 + 32);
      v10 = 59;
    }
    else
    {
      if ( v18 )
        goto LABEL_36;
      v11 = *(_QWORD *)(v11 + 32);
      v10 = 6;
    }
  }
  v12 = sub_72A270(v11, v10);
  v52 = 0;
  v51 = 0;
  v50 = sub_823970(0);
  v13 = v50;
  v14 = *(_BYTE *)(a1 + 132);
  if ( (v14 & 1) != 0 && dword_4D04880 )
  {
    if ( *v9 != 48 )
    {
      if ( !v12 || *v9 != 6 )
      {
        if ( (v14 & 0x20) != 0 )
        {
LABEL_8:
          v13 = v50;
          v15 = 0;
          v16 = 24 * v51;
          goto LABEL_9;
        }
LABEL_20:
        sub_6855B0(0xD2Fu, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
        sub_770D30(a1);
        goto LABEL_8;
      }
      v21 = *((_QWORD *)v9 + 1);
LABEL_24:
      while ( *(_BYTE *)(v21 + 140) == 12 )
        v21 = *(_QWORD *)(v21 + 160);
      if ( dword_4F077C4 == 2 )
      {
        v48 = v21;
        v39 = sub_8D23B0(v21);
        v21 = v48;
        if ( v39 )
        {
          sub_8AE000(v48);
          v21 = v48;
        }
      }
      if ( (unsigned __int8)(*(_BYTE *)(v21 + 140) - 9) <= 2u && (*(_BYTE *)(v21 + 141) & 0x20) == 0 )
      {
        v22 = *(_QWORD *)(v21 + 168);
        v23 = *(_QWORD *)(v21 + 160);
        v24 = (const __m128i **)&v50;
        v25 = *(_QWORD **)(v22 + 152);
        for ( i = v52; v23; v52 = i )
        {
          if ( v51 == i )
          {
            v40 = i;
            v44 = v24;
            sub_7A3E20(v24);
            i = v40;
            v24 = v44;
          }
          v27 = v50 + 24 * i;
          if ( v27 )
          {
            *(_BYTE *)v27 = 8;
            *(_QWORD *)(v27 + 8) = v23;
            *(_DWORD *)(v27 + 16) = 0;
          }
          v23 = *(_QWORD *)(v23 + 112);
          ++i;
        }
        v28 = v25[18];
        for ( j = v52; v28; v52 = j )
        {
          if ( v51 == j )
          {
            v41 = j;
            v45 = v24;
            sub_7A3E20(v24);
            j = v41;
            v24 = v45;
          }
          v30 = v50 + 24 * j;
          if ( v30 )
          {
            *(_BYTE *)v30 = 11;
            *(_QWORD *)(v30 + 8) = v28;
            *(_DWORD *)(v30 + 16) = 0;
          }
          v28 = *(_QWORD *)(v28 + 112);
          ++j;
        }
        v31 = v25[13];
        for ( k = v52; v31; v52 = k )
        {
          if ( v51 == k )
          {
            v42 = k;
            v46 = v24;
            sub_7A3E20(v24);
            k = v42;
            v24 = v46;
          }
          v33 = v50 + 24 * k;
          if ( v33 )
          {
            *(_BYTE *)v33 = 6;
            *(_QWORD *)(v33 + 8) = v31;
            *(_DWORD *)(v33 + 16) = 0;
          }
          v31 = *(_QWORD *)(v31 + 112);
          ++k;
        }
        v34 = v25[14];
        for ( m = v52; v34; v52 = m )
        {
          if ( v51 == m )
          {
            v43 = v34;
            v47 = v24;
            sub_7A3E20(v24);
            v34 = v43;
            v24 = v47;
          }
          v36 = v50 + 24 * m;
          if ( v36 )
          {
            *(_BYTE *)v36 = 7;
            *(_QWORD *)(v36 + 8) = v34;
            *(_DWORD *)(v36 + 16) = 0;
          }
          v34 = *(_QWORD *)(v34 + 112);
          ++m;
        }
        for ( n = *(_QWORD *)a3; *(_BYTE *)(n + 140) == 12; n = *(_QWORD *)(n + 160) )
          ;
        v38 = sub_77AFD0(a1, n, v24, (FILE *)(a3 + 28), a5, a6);
        v13 = v50;
        v15 = v38;
        v16 = 24 * v51;
        goto LABEL_9;
      }
LABEL_19:
      if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
        goto LABEL_8;
      goto LABEL_20;
    }
    v19 = *((_QWORD *)v9 + 1);
    v20 = *(_BYTE *)(v19 + 8);
    switch ( v20 )
    {
      case 1:
        *v9 = 2;
        *((_QWORD *)v9 + 1) = *(_QWORD *)(v19 + 32);
        goto LABEL_19;
      case 2:
        *v9 = 59;
        *((_QWORD *)v9 + 1) = *(_QWORD *)(v19 + 32);
        goto LABEL_19;
      case 0:
        *v9 = 6;
        v21 = *(_QWORD *)(v19 + 32);
        *((_QWORD *)v9 + 1) = v21;
        if ( !v12 )
          goto LABEL_19;
        goto LABEL_24;
    }
LABEL_36:
    sub_721090();
  }
  v16 = 0;
  v15 = 0;
  if ( (v14 & 0x20) == 0 )
  {
    sub_6855B0(0xAA1u, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
    sub_770D30(a1);
    v13 = v50;
    v16 = 24 * v51;
  }
LABEL_9:
  sub_823A00(v13, v16);
  return v15;
}
