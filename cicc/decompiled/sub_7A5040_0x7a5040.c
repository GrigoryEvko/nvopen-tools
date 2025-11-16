// Function: sub_7A5040
// Address: 0x7a5040
//
__int64 __fastcall sub_7A5040(__int64 a1, __int64 a2, __int64 a3, char **a4, __int64 a5, __int64 a6)
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
  __int64 v22; // rcx
  __int64 v23; // r13
  const __m128i **v24; // rdx
  __int64 v25; // rax
  __int64 i; // rsi
  unsigned int v27; // eax
  __int64 v28; // [rsp+8h] [rbp-68h]
  __int64 v29; // [rsp+10h] [rbp-60h]
  const __m128i **v30; // [rsp+10h] [rbp-60h]
  __int64 v32; // [rsp+20h] [rbp-50h] BYREF
  __int64 v33; // [rsp+28h] [rbp-48h]
  __int64 v34; // [rsp+30h] [rbp-40h]

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
        goto LABEL_37;
      v11 = *(_QWORD *)(v11 + 32);
      v10 = 6;
    }
  }
  v12 = sub_72A270(v11, v10);
  v34 = 0;
  v33 = 0;
  v32 = sub_823970(0);
  v13 = v32;
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
          v13 = v32;
          v15 = 0;
          v16 = 24 * v33;
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
      v29 = v21;
      if ( (unsigned int)sub_8D2870(v21) )
      {
        if ( (**(_BYTE **)(v29 + 176) & 1) == 0 )
          goto LABEL_42;
        v22 = *(_QWORD *)(v29 + 168);
        if ( (*(_BYTE *)(v29 + 161) & 0x10) != 0 )
          v22 = *(_QWORD *)(v22 + 96);
        v23 = v34;
        v24 = (const __m128i **)&v32;
        if ( v22 )
        {
          do
          {
            if ( v33 == v23 )
            {
              v28 = v22;
              v30 = v24;
              sub_7A3E20(v24);
              v22 = v28;
              v24 = v30;
            }
            v25 = v32 + 24 * v23;
            if ( v25 )
            {
              *(_BYTE *)v25 = 2;
              *(_QWORD *)(v25 + 8) = v22;
              *(_DWORD *)(v25 + 16) = 0;
            }
            v22 = *(_QWORD *)(v22 + 120);
            v34 = ++v23;
          }
          while ( v22 );
        }
        else
        {
LABEL_42:
          v24 = (const __m128i **)&v32;
        }
        for ( i = *(_QWORD *)a3; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
          ;
        v27 = sub_77AFD0(a1, i, v24, (FILE *)(a3 + 28), a5, a6);
        v13 = v32;
        v15 = v27;
        v16 = 24 * v33;
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
LABEL_37:
    sub_721090();
  }
  v16 = 0;
  v15 = 0;
  if ( (v14 & 0x20) == 0 )
  {
    sub_6855B0(0xAA1u, (FILE *)(a3 + 28), (_QWORD *)(a1 + 96));
    sub_770D30(a1);
    v13 = v32;
    v16 = 24 * v33;
  }
LABEL_9:
  sub_823A00(v13, v16);
  return v15;
}
