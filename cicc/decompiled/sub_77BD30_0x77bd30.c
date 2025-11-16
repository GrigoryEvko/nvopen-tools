// Function: sub_77BD30
// Address: 0x77bd30
//
__int64 __fastcall sub_77BD30(__int64 a1, char a2, __int64 a3, __int64 a4, unsigned int a5, __int64 a6)
{
  unsigned __int64 v9; // r15
  __int64 i; // r9
  __int64 v11; // rax
  __int64 v12; // rcx
  __int64 v13; // r8
  unsigned __int64 v14; // r9
  __int64 j; // r10
  int v16; // eax
  unsigned __int64 v17; // rcx
  unsigned __int64 v18; // r11
  char v19; // al
  char v20; // cl
  char *v21; // rbx
  char v22; // al
  size_t v23; // rdx
  __int64 v25; // r14
  int v26; // eax
  char v27; // dl
  unsigned int v28; // ecx
  int v29; // ecx
  char v30; // dl
  unsigned int v31; // ecx
  int v32; // ecx
  int v33; // eax
  unsigned int v34; // eax
  unsigned int v35; // esi
  unsigned int v36; // eax
  unsigned int v37; // eax
  unsigned int v38; // esi
  unsigned int v39; // eax
  __int64 v40; // rdx
  __int64 v41; // rdx
  int v42; // [rsp+0h] [rbp-90h]
  int v43; // [rsp+0h] [rbp-90h]
  int v44; // [rsp+0h] [rbp-90h]
  int v45; // [rsp+0h] [rbp-90h]
  int v46; // [rsp+0h] [rbp-90h]
  __int64 v47; // [rsp+8h] [rbp-88h]
  __int64 v48; // [rsp+8h] [rbp-88h]
  __int64 v49; // [rsp+8h] [rbp-88h]
  __int64 v50; // [rsp+8h] [rbp-88h]
  __int64 v51; // [rsp+8h] [rbp-88h]
  char *src; // [rsp+10h] [rbp-80h]
  char *v53; // [rsp+18h] [rbp-78h]
  char *v54; // [rsp+20h] [rbp-70h]
  __int64 v55; // [rsp+28h] [rbp-68h]
  int v56; // [rsp+28h] [rbp-68h]
  __int64 v57; // [rsp+28h] [rbp-68h]
  __int64 v58; // [rsp+30h] [rbp-60h]
  unsigned int v59; // [rsp+30h] [rbp-60h]
  __int64 v60; // [rsp+30h] [rbp-60h]
  int v63; // [rsp+40h] [rbp-50h]
  unsigned __int64 v64; // [rsp+48h] [rbp-48h]
  unsigned __int64 v65; // [rsp+48h] [rbp-48h]
  unsigned int v66; // [rsp+50h] [rbp-40h] BYREF
  unsigned int v67; // [rsp+54h] [rbp-3Ch] BYREF
  unsigned int v68; // [rsp+58h] [rbp-38h] BYREF
  _DWORD v69[13]; // [rsp+5Ch] [rbp-34h] BYREF

  v66 = 1;
  v9 = sub_777460(a1, a3);
  v64 = sub_777460(a1, a4);
  if ( !a5 )
    return v66;
  if ( !v9 || !v64 )
  {
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xC51u, (FILE *)(a6 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return 0;
    }
    return 0;
  }
  for ( i = sub_8D4130(v9); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v58 = i;
  v11 = sub_8D4130(v64);
  v14 = v58;
  for ( j = v11; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  if ( j != v58 )
  {
    v55 = j;
    v16 = sub_8D97D0(v58, j, 1, v12, v13);
    v14 = v58;
    j = v55;
    if ( !v16 )
    {
      if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
        return 0;
      sub_687430(0xC52u, a6 + 28, v58, v55, (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return 0;
    }
  }
  if ( (unsigned __int8)(*(_BYTE *)(v14 + 140) - 9) <= 2u )
  {
    v57 = j;
    v60 = v14;
    v26 = sub_8E3AD0(v14);
    v14 = v60;
    j = v57;
    if ( !v26 )
    {
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        sub_686CA0(0xC53u, a6 + 28, v60, (_QWORD *)(a1 + 96));
        sub_770D30(a1);
        return 0;
      }
      return 0;
    }
  }
  v17 = *(_QWORD *)(v14 + 128);
  if ( !v17 || (v59 = a5 / v17, v18 = a5 / v17, a5 != v18 * v17) )
  {
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xC54u, (FILE *)(a6 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return 0;
    }
    return 0;
  }
  v19 = *(_BYTE *)(a3 + 8);
  src = *(char **)a3;
  v53 = *(char **)a3;
  v20 = v19 & 8;
  if ( (v19 & 8) != 0 )
  {
    v53 = *(char **)(a3 + 16);
    if ( (v19 & 4) != 0 )
      v53 = *(char **)(*(_QWORD *)(a3 + 16) + 24LL);
  }
  v21 = *(char **)a4;
  v54 = *(char **)a4;
  if ( (*(_BYTE *)(a4 + 8) & 8) != 0 )
  {
    v54 = *(char **)(a4 + 16);
    if ( (*(_BYTE *)(a4 + 8) & 4) != 0 )
      v54 = *(char **)(*(_QWORD *)(a4 + 16) + 24LL);
  }
  v56 = 16;
  if ( (unsigned __int8)(*(_BYTE *)(v14 + 140) - 2) > 1u )
  {
    v44 = v18;
    v49 = j;
    v33 = sub_7764B0(a1, v14, &v66);
    LODWORD(v18) = v44;
    j = v49;
    v56 = v33;
    v19 = *(_BYTE *)(a3 + 8);
    v20 = v19 & 8;
  }
  if ( !v20 )
  {
    v67 = 1;
    goto LABEL_23;
  }
  v30 = *(_BYTE *)(v9 + 140);
  if ( (v19 & 1) != 0 )
  {
    v31 = 1;
    if ( v30 != 1 )
      v31 = *(_DWORD *)(v9 + 128);
    v43 = v18;
    v48 = j;
    sub_771560(a1, *(_QWORD *)(a3 + 16), v9, v31, &v67, v69, &v66);
    j = v48;
    LODWORD(v18) = v43;
    v32 = v67 - v69[0];
    goto LABEL_59;
  }
  if ( (unsigned __int8)(v30 - 2) > 1u )
  {
    v45 = v18;
    v50 = j;
    v34 = sub_7764B0(a1, v9, &v66);
    LODWORD(v18) = v45;
    v35 = v34;
    j = v50;
    if ( !v66 )
      goto LABEL_62;
    v36 = *(unsigned __int8 *)(a3 + 8);
    if ( (v36 & 8) == 0 )
    {
      v69[0] = (v36 >> 1) & 1;
      v32 = 1 - v69[0];
      goto LABEL_59;
    }
    v40 = *(_QWORD *)(a3 + 16);
    v32 = *(_DWORD *)(a3 + 8) >> 8;
    v67 = v32;
    if ( (v36 & 4) == 0 )
      goto LABEL_74;
    goto LABEL_73;
  }
  if ( v66 )
  {
    v40 = *(_QWORD *)(a3 + 16);
    v35 = 16;
    v32 = *(_DWORD *)(a3 + 8) >> 8;
    v67 = v32;
    if ( (v19 & 4) == 0 )
      goto LABEL_80;
LABEL_73:
    v40 = *(_QWORD *)(v40 + 24);
LABEL_74:
    if ( !v35 )
    {
      v69[0] = 0;
      goto LABEL_59;
    }
LABEL_80:
    v69[0] = ((unsigned int)*(_QWORD *)a3 - (unsigned int)v40) / v35;
    v32 -= v69[0];
    goto LABEL_59;
  }
LABEL_62:
  v69[0] = 0;
  v32 = 0;
LABEL_59:
  v67 = v32;
LABEL_23:
  v22 = *(_BYTE *)(a4 + 8);
  if ( (v22 & 8) == 0 )
  {
    v68 = 1;
    goto LABEL_25;
  }
  v27 = *(_BYTE *)(v64 + 140);
  if ( (v22 & 1) != 0 )
  {
    v28 = 1;
    if ( v27 != 1 )
      v28 = *(_DWORD *)(v64 + 128);
    v42 = v18;
    v47 = j;
    sub_771560(a1, *(_QWORD *)(a4 + 16), v64, v28, &v68, v69, &v66);
    j = v47;
    LODWORD(v18) = v42;
    v29 = v68 - v69[0];
    goto LABEL_54;
  }
  if ( (unsigned __int8)(v27 - 2) > 1u )
  {
    v46 = v18;
    v51 = j;
    v37 = sub_7764B0(a1, v64, &v66);
    j = v51;
    LODWORD(v18) = v46;
    v38 = v37;
    if ( !v66 )
      goto LABEL_65;
    v39 = *(unsigned __int8 *)(a4 + 8);
    if ( (v39 & 8) == 0 )
    {
      v69[0] = (v39 >> 1) & 1;
      v29 = 1 - v69[0];
      goto LABEL_54;
    }
    v41 = *(_QWORD *)(a4 + 16);
    v29 = *(_DWORD *)(a4 + 8) >> 8;
    v68 = v29;
    if ( (v39 & 4) == 0 )
      goto LABEL_78;
    goto LABEL_77;
  }
  if ( v66 )
  {
    v41 = *(_QWORD *)(a4 + 16);
    v38 = 16;
    v29 = *(_DWORD *)(a4 + 8) >> 8;
    v68 = v29;
    if ( (v22 & 4) == 0 )
      goto LABEL_81;
LABEL_77:
    v41 = *(_QWORD *)(v41 + 24);
LABEL_78:
    if ( !v38 )
    {
      v69[0] = 0;
      goto LABEL_54;
    }
LABEL_81:
    v69[0] = ((unsigned int)*(_QWORD *)a4 - (unsigned int)v41) / v38;
    v29 -= v69[0];
    goto LABEL_54;
  }
LABEL_65:
  v69[0] = 0;
  v29 = 0;
LABEL_54:
  v68 = v29;
LABEL_25:
  if ( v67 >= (unsigned int)v18 && v68 >= (unsigned int)v18 )
  {
    v23 = (unsigned int)(v18 * v56);
    if ( (a2 & 1) == 0 && v54 == v53 && (v21 > src && v21 < &src[v23] || v21 < src && src > &v21[v23]) )
    {
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        sub_6855B0(0xC56u, (FILE *)(a6 + 28), (_QWORD *)(a1 + 96));
        sub_770D30(a1);
      }
      return 0;
    }
    v63 = v18;
    v65 = j;
    memmove(v21, src, v23);
    v25 = *(_QWORD *)(a4 + 24);
    v69[0] = 0;
    if ( v63 )
    {
      do
      {
        sub_778FE0(a1, (int)v21, v65, v25);
        LODWORD(v21) = v56 + (_DWORD)v21;
        ++v69[0];
      }
      while ( v69[0] < v59 );
    }
    return v66;
  }
  if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
    return 0;
  sub_6855B0(0xC55u, (FILE *)(a6 + 28), (_QWORD *)(a1 + 96));
  sub_770D30(a1);
  return 0;
}
