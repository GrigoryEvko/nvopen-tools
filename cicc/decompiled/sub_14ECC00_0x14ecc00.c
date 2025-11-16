// Function: sub_14ECC00
// Address: 0x14ecc00
//
unsigned __int64 __fastcall sub_14ECC00(__int64 a1, char a2)
{
  unsigned int v3; // r9d
  unsigned __int64 v4; // rdi
  unsigned __int64 v5; // r10
  unsigned int v6; // r13d
  unsigned __int64 v7; // r8
  unsigned int v8; // ebx
  __int64 v9; // r15
  _QWORD *v10; // r11
  unsigned int v11; // edi
  __int64 v12; // r8
  __int64 v13; // rax
  __int64 v14; // rsi
  __int64 v15; // rdx
  char v16; // cl
  unsigned int v17; // edi
  unsigned __int64 v18; // r8
  unsigned __int64 v19; // rdx
  unsigned __int64 v20; // r15
  unsigned __int64 v22; // rdx
  unsigned int v23; // edi
  unsigned __int64 v24; // rcx
  unsigned __int64 v25; // r8
  unsigned __int64 v26; // rax
  unsigned int v27; // ebx
  unsigned __int64 v28; // r9
  unsigned __int64 *v29; // r10
  unsigned __int64 v30; // rsi
  unsigned int v31; // r13d
  unsigned __int64 v32; // rcx
  unsigned __int64 v33; // rax
  char v34; // r13
  __int64 v35; // r14
  unsigned __int64 v36; // rbx
  unsigned __int64 v37; // r9
  unsigned int v38; // r15d
  unsigned __int64 *v39; // r11
  unsigned __int64 v40; // rdi
  unsigned int v41; // r9d
  unsigned __int64 v42; // r14
  unsigned int v43; // r8d
  unsigned __int64 v44; // rdx
  unsigned int v45; // r9d
  __int64 v46; // rdx
  __int64 v47; // rsi
  char v48; // cl
  unsigned __int64 v49; // r10
  unsigned int v50; // eax
  __int64 v51; // r11
  unsigned __int64 v52; // r8
  __int64 v53; // rax
  __int64 v54; // rdx
  char v55; // cl

  while ( 1 )
  {
    v3 = *(_DWORD *)(a1 + 32);
    if ( v3 )
    {
      v8 = *(_DWORD *)(a1 + 36);
      if ( v8 <= v3 )
        goto LABEL_22;
      v4 = *(_QWORD *)(a1 + 8);
      v5 = *(_QWORD *)(a1 + 16);
      v9 = *(_QWORD *)(a1 + 24);
      v6 = v8 - v3;
      if ( v4 <= v5 )
        goto LABEL_50;
      v7 = v5 + 8;
      v10 = (_QWORD *)(v5 + *(_QWORD *)a1);
      if ( v5 + 8 > v4 )
        goto LABEL_5;
    }
    else
    {
      v4 = *(_QWORD *)(a1 + 8);
      v5 = *(_QWORD *)(a1 + 16);
      if ( v4 <= v5 )
        return 0;
      v6 = *(_DWORD *)(a1 + 36);
      if ( !v6 )
      {
        v8 = 0;
LABEL_22:
        v22 = *(_QWORD *)(a1 + 24);
        *(_DWORD *)(a1 + 32) = v3 - v8;
        v20 = v22 & (0xFFFFFFFFFFFFFFFFLL >> (64 - (unsigned __int8)v8));
        v19 = v22 >> v8;
        *(_QWORD *)(a1 + 24) = v19;
        goto LABEL_15;
      }
      v7 = v5 + 8;
      v8 = *(_DWORD *)(a1 + 36);
      v9 = 0;
      v10 = (_QWORD *)(v5 + *(_QWORD *)a1);
      if ( v5 + 8 > v4 )
      {
LABEL_5:
        *(_QWORD *)(a1 + 24) = 0;
        v11 = v4 - v5;
        if ( !v11 )
          goto LABEL_59;
        v12 = v11;
        v13 = 0;
        v14 = 0;
        do
        {
          v15 = *((unsigned __int8 *)v10 + v13);
          v16 = 8 * v13++;
          v14 |= v15 << v16;
          *(_QWORD *)(a1 + 24) = v14;
        }
        while ( v11 != v13 );
        v17 = 8 * v11;
        v7 = v5 + v12;
        goto LABEL_13;
      }
    }
    v17 = 64;
    *(_QWORD *)(a1 + 24) = *v10;
LABEL_13:
    *(_QWORD *)(a1 + 16) = v7;
    *(_DWORD *)(a1 + 32) = v17;
    if ( v6 > v17 )
      goto LABEL_50;
    v18 = *(_QWORD *)(a1 + 24);
    v19 = v18 >> v6;
    *(_QWORD *)(a1 + 24) = v18 >> v6;
    *(_DWORD *)(a1 + 32) = v3 - v8 + v17;
    v20 = ((v18 & (0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v3 - (unsigned __int8)v8 + 64))) << v3) | v9;
LABEL_15:
    if ( !(_DWORD)v20 )
      return (a2 & 1) != 0 || *(_DWORD *)(a1 + 72) && !(unsigned __int8)sub_14EB5C0(a1);
    if ( (_DWORD)v20 == 1 )
      break;
    if ( (_DWORD)v20 != 2 || (a2 & 2) != 0 )
      return ((unsigned __int64)(unsigned int)v20 << 32) | 3;
    sub_1513230(a1);
  }
  v23 = *(_DWORD *)(a1 + 32);
  if ( v23 > 7 )
  {
    LODWORD(v32) = (unsigned __int8)v19;
    *(_QWORD *)(a1 + 24) = v19 >> 8;
    *(_DWORD *)(a1 + 32) = v23 - 8;
    goto LABEL_35;
  }
  v24 = 0;
  v25 = *(_QWORD *)(a1 + 16);
  v26 = *(_QWORD *)(a1 + 8);
  if ( v23 )
    v24 = v19;
  v27 = 8 - v23;
  v28 = v24;
  if ( v25 >= v26 )
LABEL_50:
    sub_16BD130("Unexpected end of file", 1);
  v29 = (unsigned __int64 *)(v25 + *(_QWORD *)a1);
  if ( v26 >= v25 + 8 )
  {
    v30 = *v29;
    *(_QWORD *)(a1 + 16) = v25 + 8;
    v31 = 64;
    goto LABEL_34;
  }
  *(_QWORD *)(a1 + 24) = 0;
  v50 = v26 - v25;
  v51 = v50;
  v31 = 8 * v50;
  v52 = v50 + v25;
  if ( !v50 )
  {
    *(_QWORD *)(a1 + 16) = v52;
LABEL_59:
    *(_DWORD *)(a1 + 32) = 0;
    goto LABEL_50;
  }
  v53 = 0;
  v30 = 0;
  do
  {
    v54 = *((unsigned __int8 *)v29 + v53);
    v55 = 8 * v53++;
    v30 |= v54 << v55;
    *(_QWORD *)(a1 + 24) = v30;
  }
  while ( v51 != v53 );
  *(_QWORD *)(a1 + 16) = v52;
  *(_DWORD *)(a1 + 32) = v31;
  if ( v27 > v31 )
    goto LABEL_50;
LABEL_34:
  *(_QWORD *)(a1 + 24) = v30 >> v27;
  *(_DWORD *)(a1 + 32) = v23 + v31 - 8;
  v32 = v28 | (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v23 + 56)) & v30) << v23);
LABEL_35:
  v33 = (unsigned int)v32;
  if ( (v32 & 0x80) != 0 )
  {
    LODWORD(v33) = v32 & 0x7F;
    v34 = 0;
    do
    {
      v43 = *(_DWORD *)(a1 + 32);
      v34 += 7;
      if ( v43 <= 7 )
      {
        v35 = 0;
        if ( v43 )
          v35 = *(_QWORD *)(a1 + 24);
        v36 = *(_QWORD *)(a1 + 16);
        v37 = *(_QWORD *)(a1 + 8);
        v38 = 8 - v43;
        if ( v36 >= v37 )
          goto LABEL_50;
        v39 = (unsigned __int64 *)(v36 + *(_QWORD *)a1);
        if ( v37 < v36 + 8 )
        {
          *(_QWORD *)(a1 + 24) = 0;
          v45 = v37 - v36;
          if ( !v45 )
            goto LABEL_59;
          v46 = 0;
          v40 = 0;
          do
          {
            v47 = *((unsigned __int8 *)v39 + v46);
            v48 = 8 * v46++;
            v40 |= v47 << v48;
            *(_QWORD *)(a1 + 24) = v40;
          }
          while ( v45 != v46 );
          v49 = v36 + v45;
          v41 = 8 * v45;
          *(_QWORD *)(a1 + 16) = v49;
          *(_DWORD *)(a1 + 32) = v41;
          if ( v38 > v41 )
            goto LABEL_50;
        }
        else
        {
          v40 = *v39;
          *(_QWORD *)(a1 + 16) = v36 + 8;
          v41 = 64;
        }
        *(_QWORD *)(a1 + 24) = v40 >> v38;
        *(_DWORD *)(a1 + 32) = v43 + v41 - 8;
        v42 = (((0xFFFFFFFFFFFFFFFFLL >> ((unsigned __int8)v43 + 56)) & v40) << v43) | v35;
      }
      else
      {
        v44 = *(_QWORD *)(a1 + 24);
        *(_DWORD *)(a1 + 32) = v43 - 8;
        LOBYTE(v42) = v44;
        *(_QWORD *)(a1 + 24) = v44 >> 8;
      }
      v33 = ((v42 & 0x7F) << v34) | (unsigned int)v33;
    }
    while ( (v42 & 0x80) != 0 );
  }
  return (v33 << 32) | 2;
}
