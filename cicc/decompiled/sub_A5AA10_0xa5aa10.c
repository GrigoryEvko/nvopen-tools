// Function: sub_A5AA10
// Address: 0xa5aa10
//
__int64 __fastcall sub_A5AA10(__int64 a1, char *a2, __int64 a3)
{
  __int64 *v5; // r12
  unsigned __int8 v6; // al
  __int64 v7; // rbx
  __int64 v8; // rdi
  __int64 v9; // rsi
  _QWORD *v10; // rax
  const char *v11; // rsi
  __int64 result; // rax
  const char *v13; // rsi
  __int64 v14; // rbx
  __int64 v15; // rsi
  const char *v16; // rsi
  __int64 v17; // rbx
  size_t v18; // rdx
  char *v19; // rsi
  char *v20; // rbx
  unsigned __int8 v21; // si
  unsigned int v22; // ebx
  __int64 v23; // r13
  __int64 v24; // rax
  __int64 v25; // rax
  __int64 v26; // rdx
  unsigned int v27; // ebx
  __int64 v28; // r13
  __int64 v29; // rax
  unsigned int i; // ebx
  __int64 v31; // rsi
  __int64 v32; // rax
  _BYTE *v33; // rax
  _BYTE *v34; // r13
  const char *v35; // rax
  __int64 v36; // rdx
  __int64 v37; // r13
  __int64 v38; // rax
  __int64 *v39; // rbx
  __int64 v40; // rax
  int *v41; // rax
  __int64 v42; // rdx
  __int64 v43; // rbx
  __int64 v44; // r13
  _BYTE *v45; // rax
  __int64 v46; // rax
  int v47; // ecx
  unsigned int v48; // ebx
  __int64 v49; // rsi
  __int64 v50; // rax
  __int64 v51; // r13
  __int64 v52; // rax
  __int64 v53; // rdx
  __int64 v54; // rsi
  int v55; // [rsp+8h] [rbp-38h]
  int v56; // [rsp+8h] [rbp-38h]
  _BYTE *v57; // [rsp+8h] [rbp-38h]
  int v58; // [rsp+8h] [rbp-38h]

  v5 = (__int64 *)a2;
  v6 = *a2;
  if ( *a2 != 17 )
  {
    switch ( v6 )
    {
      case 0x12u:
        v14 = *((_QWORD *)a2 + 1);
        if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
        {
          sub_904010(a1, "splat (");
          v15 = v14;
          if ( (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17 <= 1 )
            v15 = **(_QWORD **)(v14 + 16);
          sub_A57EC0(*(_QWORD *)(a3 + 8), v15, a1);
          sub_904010(a1, " ");
        }
        sub_A53EB0(a1, v5 + 3);
        result = (unsigned int)*(unsigned __int8 *)(v14 + 8) - 17;
        if ( (unsigned int)result <= 1 )
          goto LABEL_14;
        return result;
      case 0xEu:
      case 0x13u:
        v13 = "zeroinitializer";
        return sub_904010(a1, v13);
      case 4u:
        sub_904010(a1, "blockaddress(");
        sub_A5A730(a1, *((_QWORD *)a2 - 8), a3);
        sub_904010(a1, ", ");
        sub_A5A730(a1, *((_QWORD *)a2 - 4), a3);
LABEL_14:
        v13 = ")";
        return sub_904010(a1, v13);
    }
    v16 = "dso_local_equivalent ";
    switch ( v6 )
    {
      case 6u:
LABEL_29:
        sub_904010(a1, v16);
        return (__int64)sub_A5A730(a1, *(v5 - 4), a3);
      case 7u:
        v16 = "no_cfi ";
        goto LABEL_29;
      case 8u:
        sub_904010(a1, "ptrauth (");
        v17 = (unsigned int)((unsigned __int8)sub_AC30F0(*(v5 - 8)) == 0) + 2;
        if ( !(unsigned __int8)sub_AC30F0(*(v5 - 4)) )
          v17 = 4;
        v18 = 0;
        v19 = 0;
        v20 = (char *)&v5[4 * v17];
        do
        {
          v5 += 4;
          sub_A51340(a1, v19, v18);
          sub_A57EC0(*(_QWORD *)(a3 + 8), *(_QWORD *)(*(v5 - 20) + 8), a1);
          sub_A51310(a1, 0x20u);
          sub_A5A730(a1, *(v5 - 20), a3);
          v19 = ", ";
          v18 = 2;
        }
        while ( v20 != (char *)v5 );
        goto LABEL_38;
      case 9u:
        v22 = 1;
        v23 = *(_QWORD *)(v5[1] + 24);
        sub_A51310(a1, 0x5Bu);
        sub_A57EC0(*(_QWORD *)(a3 + 8), v23, a1);
        sub_A51310(a1, 0x20u);
        sub_A5A730(a1, v5[-4 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)], a3);
        v55 = *((_DWORD *)v5 + 1) & 0x7FFFFFF;
        if ( v55 != 1 )
        {
          do
          {
            sub_904010(a1, ", ");
            sub_A57EC0(*(_QWORD *)(a3 + 8), v23, a1);
            sub_A51310(a1, 0x20u);
            v24 = v22++;
            sub_A5A730(a1, v5[4 * (v24 - (*((_DWORD *)v5 + 1) & 0x7FFFFFF))], a3);
          }
          while ( v22 != v55 );
        }
LABEL_43:
        v21 = 93;
        return sub_A51310(a1, v21);
      case 0xFu:
        if ( (unsigned __int8)sub_AC5570(v5, 8) )
        {
          sub_904010(a1, "c\"");
          v25 = sub_AC52D0(v5);
          sub_C92400(v25, v26, a1);
          v21 = 34;
          return sub_A51310(a1, v21);
        }
        v28 = *(_QWORD *)(v5[1] + 24);
        sub_A51310(a1, 0x5Bu);
        sub_A57EC0(*(_QWORD *)(a3 + 8), v28, a1);
        sub_A51310(a1, 0x20u);
        v29 = sub_AD68C0(v5, 0);
        sub_A5A730(a1, v29, a3);
        v56 = sub_AC5290(v5);
        if ( v56 != 1 )
        {
          for ( i = 1; i != v56; ++i )
          {
            sub_904010(a1, ", ");
            sub_A57EC0(*(_QWORD *)(a3 + 8), v28, a1);
            sub_A51310(a1, 0x20u);
            v31 = i;
            v32 = sub_AD68C0(v5, v31);
            sub_A5A730(a1, v32, a3);
          }
        }
        goto LABEL_43;
      case 0xAu:
        if ( (*(_BYTE *)(v5[1] + 9) & 2) != 0 )
          sub_A51310(a1, 0x3Cu);
        sub_A51310(a1, 0x7Bu);
        v27 = *((_DWORD *)v5 + 1) & 0x7FFFFFF;
        if ( v27 )
        {
          v51 = 1;
          sub_A51310(a1, 0x20u);
          sub_A57EC0(*(_QWORD *)(a3 + 8), *(_QWORD *)(v5[-4 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)] + 8), a1);
          sub_A51310(a1, 0x20u);
          sub_A5A730(a1, v5[-4 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)], a3);
          if ( v27 != 1 )
          {
            do
            {
              sub_904010(a1, ", ");
              sub_A57EC0(*(_QWORD *)(a3 + 8), *(_QWORD *)(v5[4 * (v51 - (*((_DWORD *)v5 + 1) & 0x7FFFFFF))] + 8), a1);
              sub_A51310(a1, 0x20u);
              v52 = v51++;
              sub_A5A730(a1, v5[4 * (v52 - (*((_DWORD *)v5 + 1) & 0x7FFFFFF))], a3);
            }
            while ( v27 > (unsigned int)v51 );
          }
          sub_A51310(a1, 0x20u);
        }
        sub_A51310(a1, 0x7Du);
        result = v5[1];
        if ( (*(_BYTE *)(result + 9) & 2) == 0 )
          return result;
        goto LABEL_52;
      case 0x10u:
      case 0xBu:
        v43 = v5[1];
        v44 = *(_QWORD *)(v43 + 24);
        v45 = (_BYTE *)sub_AD7630(v5, 0);
        if ( !v45 || (v57 = v45, (unsigned __int8)(*v45 - 17) > 1u) )
        {
          sub_A51310(a1, 0x3Cu);
          sub_A57EC0(*(_QWORD *)(a3 + 8), v44, a1);
          sub_A51310(a1, 0x20u);
          v46 = sub_AD69F0(v5, 0);
          sub_A5A730(a1, v46, a3);
          v47 = *(_DWORD *)(v43 + 32);
          v48 = 1;
          if ( v47 != 1 )
          {
            do
            {
              v58 = v47;
              sub_904010(a1, ", ");
              sub_A57EC0(*(_QWORD *)(a3 + 8), v44, a1);
              sub_A51310(a1, 0x20u);
              v49 = v48++;
              v50 = sub_AD69F0(v5, v49);
              sub_A5A730(a1, v50, a3);
              v47 = v58;
            }
            while ( v48 != v58 );
          }
LABEL_52:
          v21 = 62;
          return sub_A51310(a1, v21);
        }
        sub_904010(a1, "splat (");
        sub_A57EC0(*(_QWORD *)(a3 + 8), v44, a1);
        sub_A51310(a1, 0x20u);
        v53 = a3;
        v54 = (__int64)v57;
        break;
      case 0x14u:
        v13 = "null";
        return sub_904010(a1, v13);
      default:
        v13 = "none";
        if ( v6 == 21 )
          return sub_904010(a1, v13);
        v13 = "poison";
        if ( v6 == 13 )
          return sub_904010(a1, v13);
        v13 = "undef";
        if ( (unsigned int)v6 - 12 <= 1 )
          return sub_904010(a1, v13);
        if ( v6 != 5 )
        {
          v13 = "<placeholder or erroneous Constant>";
          return sub_904010(a1, v13);
        }
        if ( *((_WORD *)v5 + 1) != 63
          || (v13 = 0, v33 = (_BYTE *)sub_AD7630(v5, 0), (v34 = v33) == 0)
          || (unsigned __int8)(*v33 - 17) > 1u )
        {
          v35 = (const char *)sub_AC4870(v5, v13);
          sub_904010(a1, v35);
          sub_A52910(a1, (unsigned __int8 *)v5);
          sub_904010(a1, " (");
          if ( *((_WORD *)v5 + 1) == 34 )
          {
            v37 = *(_QWORD *)(a3 + 8);
            v38 = sub_BB5290(v5, " (", v36);
            sub_A57EC0(v37, v38, a1);
            sub_904010(a1, ", ");
          }
          v39 = &v5[-4 * (*((_DWORD *)v5 + 1) & 0x7FFFFFF)];
          while ( v39 != v5 )
          {
            v40 = *v39;
            v39 += 4;
            sub_A57EC0(*(_QWORD *)(a3 + 8), *(_QWORD *)(v40 + 8), a1);
            sub_A51310(a1, 0x20u);
            sub_A5A730(a1, *(v39 - 4), a3);
            if ( v39 == v5 )
              break;
            sub_904010(a1, ", ");
          }
          if ( (unsigned __int8)sub_AC35E0(v5) )
          {
            sub_904010(a1, " to ");
            sub_A57EC0(*(_QWORD *)(a3 + 8), v5[1], a1);
          }
          if ( *((_WORD *)v5 + 1) == 63 )
          {
            v41 = (int *)sub_AC35F0(v5);
            sub_A52400(a1, v5[1], v41, v42);
          }
          goto LABEL_38;
        }
        sub_904010(a1, "splat (");
        sub_A57EC0(*(_QWORD *)(a3 + 8), *((_QWORD *)v34 + 1), a1);
        sub_A51310(a1, 0x20u);
        v53 = a3;
        v54 = (__int64)v34;
        break;
    }
    sub_A5A730(a1, v54, v53);
LABEL_38:
    v21 = 41;
    return sub_A51310(a1, v21);
  }
  v7 = *((_QWORD *)a2 + 1);
  v8 = v7;
  if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
  {
    sub_904010(a1, "splat (");
    v9 = v7;
    if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 <= 1 )
      v9 = **(_QWORD **)(v7 + 16);
    sub_A57EC0(*(_QWORD *)(a3 + 8), v9, a1);
    sub_904010(a1, " ");
    if ( (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17 > 1 )
      v8 = v7;
    else
      v8 = **(_QWORD **)(v7 + 16);
  }
  if ( (unsigned __int8)sub_BCAC40(v8, 1) )
  {
    v10 = (_QWORD *)v5[3];
    if ( *((_DWORD *)v5 + 8) > 0x40u )
      v10 = (_QWORD *)*v10;
    v11 = "true";
    if ( !v10 )
      v11 = "false";
    sub_904010(a1, v11);
  }
  else
  {
    sub_C49420(v5 + 3, a1, 1);
  }
  result = (unsigned int)*(unsigned __int8 *)(v7 + 8) - 17;
  if ( (unsigned int)result <= 1 )
    goto LABEL_14;
  return result;
}
