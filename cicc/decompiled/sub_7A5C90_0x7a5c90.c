// Function: sub_7A5C90
// Address: 0x7a5c90
//
__int64 __fastcall sub_7A5C90(__int64 a1, __int64 a2, __int64 a3, char **a4, __int64 a5, __int64 a6)
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
  __int64 i; // rcx
  __int64 v25; // rax
  __int64 v26; // rdx
  __int64 j; // r12
  __int64 v28; // rax
  __int64 k; // rsi
  unsigned int v30; // eax
  int v31; // eax
  __int64 v32; // [rsp+0h] [rbp-70h]
  __int64 v33; // [rsp+8h] [rbp-68h]
  __int64 v34; // [rsp+8h] [rbp-68h]
  const __m128i **v35; // [rsp+10h] [rbp-60h]
  const __m128i **v36; // [rsp+10h] [rbp-60h]
  __int64 v37; // [rsp+10h] [rbp-60h]
  __int64 v39; // [rsp+20h] [rbp-50h] BYREF
  __int64 v40; // [rsp+28h] [rbp-48h]
  __int64 v41; // [rsp+30h] [rbp-40h]

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
  v41 = 0;
  v40 = 0;
  v39 = sub_823970(0);
  v13 = v39;
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
          v13 = v39;
          v15 = 0;
          v16 = 24 * v40;
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
        v37 = v21;
        v31 = sub_8D23B0(v21);
        v21 = v37;
        if ( v31 )
        {
          sub_8AE000(v37);
          v21 = v37;
        }
      }
      if ( (unsigned __int8)(*(_BYTE *)(v21 + 140) - 9) <= 2u && (*(_BYTE *)(v21 + 141) & 0x20) == 0 )
      {
        v22 = v41;
        v23 = (const __m128i **)&v39;
        for ( i = *(_QWORD *)(*(_QWORD *)(v21 + 168) + 8LL); i; i = *(_QWORD *)(i + 8) )
        {
          if ( v40 == v22 )
          {
            v32 = v21;
            v33 = i;
            v35 = v23;
            sub_7A3E20(v23);
            v21 = v32;
            i = v33;
            v23 = v35;
          }
          v25 = v39 + 24 * v22;
          if ( v25 )
          {
            *(_BYTE *)v25 = 37;
            *(_QWORD *)(v25 + 8) = i;
            *(_DWORD *)(v25 + 16) = 0;
          }
          v41 = ++v22;
        }
        v26 = *(_QWORD *)(v21 + 160);
        for ( j = v41; v26; v41 = j )
        {
          if ( v40 == j )
          {
            v34 = v26;
            v36 = v23;
            sub_7A3E20(v23);
            v26 = v34;
            v23 = v36;
          }
          v28 = v39 + 24 * j;
          if ( v28 )
          {
            *(_BYTE *)v28 = 8;
            *(_QWORD *)(v28 + 8) = v26;
            *(_DWORD *)(v28 + 16) = 0;
          }
          v26 = *(_QWORD *)(v26 + 112);
          ++j;
        }
        for ( k = *(_QWORD *)a3; *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
          ;
        v30 = sub_77AFD0(a1, k, v23, (FILE *)(a3 + 28), a5, a6);
        v13 = v39;
        v15 = v30;
        v16 = 24 * v40;
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
    v13 = v39;
    v16 = 24 * v40;
  }
LABEL_9:
  sub_823A00(v13, v16);
  return v15;
}
