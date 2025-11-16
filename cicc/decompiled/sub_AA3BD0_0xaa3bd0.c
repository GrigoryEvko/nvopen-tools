// Function: sub_AA3BD0
// Address: 0xaa3bd0
//
char *__fastcall sub_AA3BD0(_QWORD *a1)
{
  char *v2; // r13
  char *v3; // r12
  __int64 v4; // rax
  char *v5; // rax
  __int64 v6; // rsi
  __int64 v7; // rsi
  __int64 v8; // rsi
  __int64 v9; // rsi
  _QWORD *i; // rbx
  _BYTE *v11; // rdi
  _BYTE *v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rdi
  __int64 v15; // rsi
  size_t v16; // rdx
  __int64 v17; // rax
  _QWORD *v19; // rax
  _QWORD *v20; // rax
  _QWORD *v21; // rax

  v2 = (char *)a1[1];
  v3 = (char *)*a1;
  v4 = 0x6DB6DB6DB6DB6DB7LL * ((__int64)&v2[-*a1] >> 3);
  if ( v4 >> 2 <= 0 )
  {
LABEL_47:
    if ( v4 == 2 )
    {
      if ( *((_QWORD *)v3 + 1) != 22 )
        goto LABEL_59;
      v21 = *(_QWORD **)v3;
    }
    else
    {
      if ( v4 != 3 )
      {
        if ( v4 != 1 || *((_QWORD *)v3 + 1) != 22 )
          goto LABEL_50;
        v20 = *(_QWORD **)v3;
        goto LABEL_61;
      }
      if ( *((_QWORD *)v3 + 1) == 22 )
      {
        v19 = *(_QWORD **)v3;
        if ( !(**(_QWORD **)v3 ^ 0x72612E676E616C63LL | *(_QWORD *)(*(_QWORD *)v3 + 8LL) ^ 0x6863617474612E63LL)
          && *((_DWORD *)v19 + 4) == 1633903717
          && *((_WORD *)v19 + 10) == 27756
          && *((_QWORD *)v3 + 4) == *((_QWORD *)v3 + 5) )
        {
          goto LABEL_27;
        }
      }
      if ( *((_QWORD *)v3 + 8) != 22 )
      {
        v3 += 56;
LABEL_59:
        if ( *((_QWORD *)v3 + 8) != 22 )
          goto LABEL_50;
        v20 = (_QWORD *)*((_QWORD *)v3 + 7);
        v3 += 56;
LABEL_61:
        if ( !(*v20 ^ 0x72612E676E616C63LL | v20[1] ^ 0x6863617474612E63LL)
          && *((_DWORD *)v20 + 4) == 1633903717
          && *((_WORD *)v20 + 10) == 27756
          && *((_QWORD *)v3 + 4) == *((_QWORD *)v3 + 5) )
        {
          goto LABEL_27;
        }
LABEL_50:
        v3 = (char *)a1[1];
        return sub_AA39D0((__int64)a1, v3, v2);
      }
      v21 = (_QWORD *)*((_QWORD *)v3 + 7);
      v3 += 56;
    }
    if ( !(*v21 ^ 0x72612E676E616C63LL | v21[1] ^ 0x6863617474612E63LL)
      && *((_DWORD *)v21 + 4) == 1633903717
      && *((_WORD *)v21 + 10) == 27756
      && *((_QWORD *)v3 + 4) == *((_QWORD *)v3 + 5) )
    {
      goto LABEL_27;
    }
    goto LABEL_59;
  }
  v5 = &v3[224 * (v4 >> 2)];
  while ( 1 )
  {
    if ( *((_QWORD *)v3 + 1) == 22 )
    {
      v6 = *(_QWORD *)v3;
      if ( !(**(_QWORD **)v3 ^ 0x72612E676E616C63LL | *(_QWORD *)(*(_QWORD *)v3 + 8LL) ^ 0x6863617474612E63LL)
        && *(_DWORD *)(v6 + 16) == 1633903717
        && *(_WORD *)(v6 + 20) == 27756
        && *((_QWORD *)v3 + 4) == *((_QWORD *)v3 + 5) )
      {
        goto LABEL_27;
      }
    }
    if ( *((_QWORD *)v3 + 8) == 22 )
    {
      v7 = *((_QWORD *)v3 + 7);
      if ( !(*(_QWORD *)v7 ^ 0x72612E676E616C63LL | *(_QWORD *)(v7 + 8) ^ 0x6863617474612E63LL)
        && *(_DWORD *)(v7 + 16) == 1633903717
        && *(_WORD *)(v7 + 20) == 27756
        && *((_QWORD *)v3 + 11) == *((_QWORD *)v3 + 12) )
      {
        v3 += 56;
        goto LABEL_27;
      }
    }
    if ( *((_QWORD *)v3 + 15) == 22 )
    {
      v8 = *((_QWORD *)v3 + 14);
      if ( !(*(_QWORD *)v8 ^ 0x72612E676E616C63LL | *(_QWORD *)(v8 + 8) ^ 0x6863617474612E63LL)
        && *(_DWORD *)(v8 + 16) == 1633903717
        && *(_WORD *)(v8 + 20) == 27756
        && *((_QWORD *)v3 + 18) == *((_QWORD *)v3 + 19) )
      {
        break;
      }
    }
    if ( *((_QWORD *)v3 + 22) == 22 )
    {
      v9 = *((_QWORD *)v3 + 21);
      if ( !(*(_QWORD *)v9 ^ 0x72612E676E616C63LL | *(_QWORD *)(v9 + 8) ^ 0x6863617474612E63LL)
        && *(_DWORD *)(v9 + 16) == 1633903717
        && *(_WORD *)(v9 + 20) == 27756
        && *((_QWORD *)v3 + 25) == *((_QWORD *)v3 + 26) )
      {
        v3 += 168;
        goto LABEL_27;
      }
    }
    v3 += 224;
    if ( v3 == v5 )
    {
      v4 = 0x6DB6DB6DB6DB6DB7LL * ((v2 - v3) >> 3);
      goto LABEL_47;
    }
  }
  v3 += 112;
LABEL_27:
  if ( v2 != v3 && v2 != v3 + 56 )
  {
    for ( i = v3 + 72; ; i += 7 )
    {
      v16 = *(i - 1);
      v17 = *(i - 2);
      if ( v16 != 22 )
        break;
      if ( *(_QWORD *)v17 ^ 0x72612E676E616C63LL | *(_QWORD *)(v17 + 8) ^ 0x6863617474612E63LL
        || *(_DWORD *)(v17 + 16) != 1633903717
        || *(_WORD *)(v17 + 20) != 27756 )
      {
        v11 = *(_BYTE **)v3;
        if ( (_QWORD *)v17 == i )
          goto LABEL_80;
LABEL_33:
        v12 = v3 + 16;
        goto LABEL_34;
      }
      if ( i[2] == i[3] )
        goto LABEL_40;
      v11 = *(_BYTE **)v3;
      v12 = v3 + 16;
      if ( (_QWORD *)v17 == i )
        goto LABEL_80;
LABEL_34:
      if ( v11 == v12 )
      {
        *(_QWORD *)v3 = v17;
        *((_QWORD *)v3 + 1) = *(i - 1);
        *((_QWORD *)v3 + 2) = *i;
LABEL_76:
        *(i - 2) = i;
        v11 = i;
        goto LABEL_37;
      }
      *(_QWORD *)v3 = v17;
      v13 = *((_QWORD *)v3 + 2);
      *((_QWORD *)v3 + 1) = *(i - 1);
      *((_QWORD *)v3 + 2) = *i;
      if ( !v11 )
        goto LABEL_76;
      *(i - 2) = v11;
      *i = v13;
LABEL_37:
      *(i - 1) = 0;
      *v11 = 0;
      v14 = *((_QWORD *)v3 + 4);
      v15 = *((_QWORD *)v3 + 6);
      *((_QWORD *)v3 + 4) = i[2];
      *((_QWORD *)v3 + 5) = i[3];
      *((_QWORD *)v3 + 6) = i[4];
      i[2] = 0;
      i[3] = 0;
      i[4] = 0;
      if ( v14 )
        j_j___libc_free_0(v14, v15 - v14);
      v3 += 56;
LABEL_40:
      if ( v2 == (char *)(i + 5) )
        return sub_AA39D0((__int64)a1, v3, v2);
    }
    v11 = *(_BYTE **)v3;
    if ( (_QWORD *)v17 == i )
    {
      if ( v16 )
      {
        if ( v16 == 1 )
        {
          *v11 = *(_BYTE *)i;
          v16 = *(i - 1);
          v11 = *(_BYTE **)v3;
        }
        else
        {
LABEL_80:
          memcpy(v11, i, v16);
          v16 = *(i - 1);
          v11 = *(_BYTE **)v3;
        }
      }
      *((_QWORD *)v3 + 1) = v16;
      v11[v16] = 0;
      v11 = (_BYTE *)*(i - 2);
      goto LABEL_37;
    }
    goto LABEL_33;
  }
  return sub_AA39D0((__int64)a1, v3, v2);
}
