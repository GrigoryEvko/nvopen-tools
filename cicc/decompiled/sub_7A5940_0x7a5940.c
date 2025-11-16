// Function: sub_7A5940
// Address: 0x7a5940
//
__int64 __fastcall sub_7A5940(__int64 a1, __int64 a2, __int64 a3, char **a4, __int64 a5, __int64 a6)
{
  char *v9; // r13
  char v10; // si
  __int64 v11; // rdi
  __int64 v12; // r12
  __int64 v13; // rdi
  char v14; // al
  unsigned int v15; // r12d
  __int64 v16; // rsi
  char v18; // al
  __int64 v19; // rdx
  char v20; // al
  __int64 v21; // rdx
  __int64 v22; // r12
  const __m128i **v23; // r10
  __int64 i; // rdx
  __int64 v25; // rax
  __int64 j; // rsi
  unsigned int v27; // eax
  int v28; // eax
  __int64 v29; // [rsp+8h] [rbp-68h]
  const __m128i **v30; // [rsp+10h] [rbp-60h]
  __int64 v31; // [rsp+10h] [rbp-60h]
  __int64 v33; // [rsp+20h] [rbp-50h] BYREF
  __int64 v34; // [rsp+28h] [rbp-48h]
  __int64 v35; // [rsp+30h] [rbp-40h]

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
  v35 = 0;
  v34 = 0;
  v33 = sub_823970(0);
  v13 = v33;
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
          v13 = v33;
          v15 = 0;
          v16 = 24 * v34;
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
        v31 = v21;
        v28 = sub_8D23B0(v21);
        v21 = v31;
        if ( v28 )
        {
          sub_8AE000(v31);
          v21 = v31;
        }
      }
      if ( (unsigned __int8)(*(_BYTE *)(v21 + 140) - 9) <= 2u && (*(_BYTE *)(v21 + 141) & 0x20) == 0 )
      {
        v22 = v35;
        v23 = (const __m128i **)&v33;
        for ( i = *(_QWORD *)(*(_QWORD *)(v21 + 168) + 8LL); i; i = *(_QWORD *)(i + 8) )
        {
          if ( v34 == v22 )
          {
            v29 = i;
            v30 = v23;
            sub_7A3E20(v23);
            i = v29;
            v23 = v30;
          }
          v25 = v33 + 24 * v22;
          if ( v25 )
          {
            *(_BYTE *)v25 = 37;
            *(_QWORD *)(v25 + 8) = i;
            *(_DWORD *)(v25 + 16) = 0;
          }
          v35 = ++v22;
        }
        for ( j = *(_QWORD *)a3; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        v27 = sub_77AFD0(a1, j, v23, (FILE *)(a3 + 28), a5, a6);
        v13 = v33;
        v15 = v27;
        v16 = 24 * v34;
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
    v13 = v33;
    v16 = 24 * v34;
  }
LABEL_9:
  sub_823A00(v13, v16);
  return v15;
}
