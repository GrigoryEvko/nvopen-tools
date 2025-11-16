// Function: sub_82D8D0
// Address: 0x82d8d0
//
void __fastcall sub_82D8D0(__int64 *a1, __int64 a2, _QWORD *a3, _DWORD *a4, __int64 a5, __int64 a6)
{
  __int64 v6; // r12
  unsigned __int64 v7; // rcx
  _QWORD *v8; // rbx
  _QWORD *v9; // r13
  _QWORD *v10; // r14
  __int64 *v11; // rax
  _BYTE *v12; // rax
  __int64 v13; // rax
  bool v14; // zf
  int v15; // eax
  __int64 v16; // r15
  __int64 v17; // rbx
  __int64 v18; // r12
  __int64 j; // r15
  __int64 v20; // r14
  __int64 v21; // r13
  __int64 v22; // rax
  int v23; // eax
  __int64 v24; // rax
  char v25; // dl
  unsigned __int64 v26; // r14
  __int64 v27; // r13
  __int64 v28; // rbx
  int v29; // eax
  _QWORD *v30; // rbx
  __int64 v31; // rax
  __int64 v32; // r13
  unsigned __int64 v33; // rax
  __int64 v34; // rsi
  __int64 v35; // rsi
  __int64 v36; // r15
  __int64 v37; // rax
  _QWORD *v38; // r12
  char *v39; // r13
  __int64 v40; // rax
  __int64 v41; // r13
  _QWORD *v42; // r15
  __int64 v43; // rbx
  __int64 k; // r13
  __int64 v45; // r15
  __int64 v46; // rsi
  __int64 v47; // rbx
  __int64 *v48; // rax
  __int64 v49; // rax
  int v50; // edi
  int v51; // esi
  int v52; // eax
  __int64 v53; // rax
  _QWORD *v54; // [rsp+0h] [rbp-80h]
  __int64 v55; // [rsp+8h] [rbp-78h]
  _DWORD *v58; // [rsp+20h] [rbp-60h]
  int v59; // [rsp+28h] [rbp-58h]
  __int64 v60; // [rsp+28h] [rbp-58h]
  int v61; // [rsp+30h] [rbp-50h]
  unsigned int v62; // [rsp+34h] [rbp-4Ch]
  int v63; // [rsp+34h] [rbp-4Ch]
  unsigned __int64 i; // [rsp+40h] [rbp-40h]
  unsigned int v66; // [rsp+48h] [rbp-38h]
  int v67; // [rsp+48h] [rbp-38h]

  v6 = *a1;
  *(_DWORD *)a3 = 0;
  v58 = a3;
  *a4 = 0;
  if ( !v6 )
    return;
  v7 = *(_QWORD *)v6;
  if ( !*(_QWORD *)v6 )
    goto LABEL_128;
  v8 = (_QWORD *)v6;
  v62 = 0;
  v9 = 0;
  v10 = *(_QWORD **)v6;
  v11 = *(__int64 **)(v6 + 120);
  v66 = 0;
  v61 = 0;
  for ( i = 0; v11; v11 = (__int64 *)*v11 )
  {
LABEL_6:
    if ( *((_BYTE *)v11 + 12) )
    {
      v12 = (_BYTE *)v8[6];
      if ( v12 && *v12 == 69 )
      {
        v59 = 1;
        goto LABEL_17;
      }
      if ( v66 )
        goto LABEL_10;
      if ( v62 )
      {
        if ( (*((_BYTE *)v8 + 145) & 0x10) != 0 )
          goto LABEL_39;
        goto LABEL_10;
      }
      goto LABEL_32;
    }
  }
  while ( 1 )
  {
    v12 = (_BYTE *)v8[6];
    if ( !v12 || (v59 = 0, *v12 != 69) )
    {
      v7 = v62;
      if ( !v62 )
        goto LABEL_29;
      if ( (*((_BYTE *)v8 + 145) & 0x10) != 0 )
        goto LABEL_36;
      goto LABEL_10;
    }
LABEL_17:
    if ( v12[1] == 69 )
    {
      v36 = v6;
      if ( v6 )
      {
        v55 = v6;
        v54 = v9;
        do
        {
          if ( (_QWORD *)v36 != v8 )
          {
            v37 = *(_QWORD *)(v36 + 8);
            if ( v37 )
            {
              if ( !*(_BYTE *)(v36 + 32) && (*(_BYTE *)(v37 + 81) & 0x10) == 0 )
              {
                v38 = *(_QWORD **)(v36 + 120);
                v39 = (char *)v8[6];
                if ( !v38 )
                {
LABEL_127:
                  v6 = v55;
                  v9 = v54;
                  goto LABEL_10;
                }
                while ( (unsigned int)sub_827590(v38[4], *v39) )
                {
                  v38 = (_QWORD *)*v38;
                  ++v39;
                  if ( !v38 )
                    goto LABEL_127;
                }
              }
            }
          }
          v36 = *(_QWORD *)v36;
        }
        while ( v36 );
        v6 = v55;
        v9 = v54;
      }
    }
    if ( (v66 & v59) != 0 )
      goto LABEL_10;
    if ( !v62 )
      break;
    if ( (*((_BYTE *)v8 + 145) & 0x10) != 0 )
    {
      if ( v59 )
        goto LABEL_39;
LABEL_36:
      if ( !v9 || (a3 = (_QWORD *)v66, v66) )
      {
        v66 = 1;
        goto LABEL_39;
      }
      goto LABEL_135;
    }
LABEL_10:
    if ( v9 )
    {
      *v9 = v10;
    }
    else
    {
      v6 = (__int64)v10;
      *a1 = (__int64)v10;
    }
    *v8 = 0;
    sub_725130((__int64 *)v8[5]);
    sub_82D8A0((_QWORD *)v8[15]);
    *v8 = qword_4D03C68;
    qword_4D03C68 = v8;
LABEL_13:
    if ( !v10 )
      goto LABEL_46;
LABEL_14:
    v8 = v10;
    v10 = (_QWORD *)*v10;
    v11 = (__int64 *)v8[15];
    if ( v11 )
      goto LABEL_6;
  }
  if ( v59 )
    goto LABEL_32;
LABEL_29:
  if ( !v9 || v66 )
  {
    v66 = 1;
    goto LABEL_32;
  }
LABEL_135:
  *v9 = 0;
  if ( v6 )
  {
    v42 = v8;
    do
    {
      v43 = v6;
      v6 = *(_QWORD *)v6;
      sub_725130(*(__int64 **)(v43 + 40));
      sub_82D8A0(*(_QWORD **)(v43 + 120));
      a3 = qword_4D03C68;
      *(_QWORD *)v43 = qword_4D03C68;
      qword_4D03C68 = (_QWORD *)v43;
    }
    while ( v6 );
    v8 = v42;
  }
  v66 = 1;
  v6 = (__int64)v8;
  i = 0;
  *a1 = (__int64)v8;
LABEL_32:
  if ( (*((_BYTE *)v8 + 145) & 0x10) == 0 )
  {
    ++i;
    goto LABEL_42;
  }
LABEL_39:
  if ( v9 && !v62 )
  {
    for ( *v9 = 0; v6; qword_4D03C68 = (_QWORD *)v41 )
    {
      v41 = v6;
      v6 = *(_QWORD *)v6;
      sub_725130(*(__int64 **)(v41 + 40));
      sub_82D8A0(*(_QWORD **)(v41 + 120));
      *(_QWORD *)v41 = qword_4D03C68;
    }
    i = 1;
    v6 = (__int64)v8;
    v62 = 1;
    *a1 = (__int64)v8;
  }
  else
  {
    ++i;
    v62 = 1;
  }
LABEL_42:
  v9 = v8;
  *((_BYTE *)v8 + 146) = *((_BYTE *)v8 + 146) & 0xFC | 1;
  v8[16] = v8[15];
  v13 = v8[1];
  if ( !v13 )
    goto LABEL_13;
  v14 = (*(_BYTE *)(v13 + 82) & 4) == 0;
  v15 = 1;
  if ( v14 )
    v15 = v61;
  v61 = v15;
  if ( v10 )
    goto LABEL_14;
LABEL_46:
  if ( v61 )
  {
    if ( !i )
    {
      *a1 = 0;
      if ( !v6 )
        goto LABEL_98;
      goto LABEL_86;
    }
LABEL_207:
    *a1 = 0;
    if ( !v6 )
    {
      if ( *v58 )
        goto LABEL_26;
      return;
    }
    goto LABEL_87;
  }
  if ( i == 1 )
    goto LABEL_207;
  v16 = *(_QWORD *)(v6 + 128);
  v63 = 0;
  if ( v16 )
  {
    v60 = v6;
    while ( 1 )
    {
      v17 = 0;
      v18 = v16;
      for ( j = v60; ; v18 = *(_QWORD *)(j + 128) )
      {
        if ( *(_DWORD *)(v18 + 8) == 6 )
        {
          v63 = 1;
          goto LABEL_52;
        }
        if ( v17 )
        {
          v67 = 0;
          v20 = 0;
          v21 = v17;
          do
          {
            while ( 1 )
            {
              v23 = sub_828DA0(v18, *(_QWORD *)(v21 + 128));
              if ( v23 >= 0 )
                break;
              v20 = v21;
              v67 = 1;
              v21 = *(_QWORD *)(v21 + 136);
              if ( !v21 )
                goto LABEL_63;
            }
            if ( v23 )
            {
              *(_BYTE *)(v21 + 146) &= ~4u;
              v22 = *(_QWORD *)(v21 + 136);
              if ( v20 )
              {
                *(_QWORD *)(v20 + 136) = v22;
                v22 = *(_QWORD *)(v21 + 136);
              }
              else
              {
                v17 = *(_QWORD *)(v21 + 136);
              }
            }
            else
            {
              v22 = *(_QWORD *)(v21 + 136);
              v20 = v21;
            }
            v21 = v22;
          }
          while ( v22 );
LABEL_63:
          if ( v67 )
            break;
        }
LABEL_52:
        *(_QWORD *)(j + 136) = v17;
        v17 = j;
        *(_BYTE *)(j + 146) |= 4u;
        j = *(_QWORD *)j;
        if ( !j )
          goto LABEL_65;
LABEL_53:
        ;
      }
      *(_BYTE *)(j + 146) &= ~4u;
      j = *(_QWORD *)j;
      if ( j )
        goto LABEL_53;
LABEL_65:
      v24 = v60;
      v7 = i;
      do
      {
        v25 = *(_BYTE *)(v24 + 146);
        if ( (v25 & 4) != 0 )
        {
          *(_BYTE *)(v24 + 146) = v25 | 2;
        }
        else if ( (v25 & 1) != 0 )
        {
          --v7;
          *(_BYTE *)(v24 + 146) = v25 & 0xFE;
        }
        a3 = **(_QWORD ***)(v24 + 128);
        *(_QWORD *)(v24 + 128) = a3;
        v24 = *(_QWORD *)v24;
      }
      while ( v24 );
      i = v7;
      v16 = *(_QWORD *)(v60 + 128);
      if ( !v16 )
      {
        v6 = v60;
        if ( v7 > 1 )
          goto LABEL_75;
        if ( v7 == 1 )
          goto LABEL_144;
LABEL_141:
        if ( dword_4F077BC )
        {
          i = 0;
          goto LABEL_176;
        }
        *a1 = 0;
LABEL_86:
        i = 0;
        v61 = 1;
        goto LABEL_87;
      }
    }
  }
  if ( i <= 1 )
    goto LABEL_141;
LABEL_75:
  v26 = i;
  v27 = 0;
  v28 = v6;
  while ( 2 )
  {
    while ( 2 )
    {
      if ( (*(_BYTE *)(v28 + 146) & 1) == 0 )
      {
LABEL_78:
        v28 = *(_QWORD *)v28;
        if ( !v28 )
          goto LABEL_82;
        continue;
      }
      break;
    }
    if ( v27 )
    {
      v29 = sub_82A120(v28, v27);
      if ( v29 < 0 )
      {
        --v26;
        *(_BYTE *)(v28 + 146) &= ~1u;
        if ( v26 == 1 )
          goto LABEL_144;
      }
      else if ( v29 )
      {
        if ( v6 != v28 )
        {
          v49 = v6;
          while ( 1 )
          {
            a3 = (_QWORD *)*(unsigned __int8 *)(v49 + 146);
            if ( ((unsigned __int8)a3 & 1) != 0 )
            {
              a3 = (_QWORD *)((unsigned int)a3 & 0xFFFFFFFE);
              --v26;
              *(_BYTE *)(v49 + 146) = (_BYTE)a3;
              if ( v26 == 1 )
                break;
            }
            v49 = *(_QWORD *)v49;
            if ( v49 == v28 )
              goto LABEL_81;
          }
LABEL_144:
          for ( k = v6; (*(_BYTE *)(k + 146) & 1) == 0; k = *(_QWORD *)k )
            ;
          v45 = v6;
          while ( 1 )
          {
            if ( k != v45 )
            {
              *(_QWORD *)(k + 128) = *(_QWORD *)(k + 120);
              v46 = *(_QWORD *)(v45 + 120);
              *(_QWORD *)(v45 + 128) = v46;
              v47 = *(_QWORD *)(k + 128);
              if ( v47 )
              {
                while ( (int)sub_828DA0(v47, v46) <= 0 && *(_DWORD *)(v47 + 8) != 6 )
                {
                  *(_QWORD *)(k + 128) = **(_QWORD **)(k + 128);
                  v48 = *(__int64 **)(v45 + 128);
                  v46 = *v48;
                  *(_QWORD *)(v45 + 128) = *v48;
                  v47 = *(_QWORD *)(k + 128);
                  if ( !v47 )
                    goto LABEL_153;
                }
              }
              else
              {
LABEL_153:
                if ( (int)sub_82A120(k, v45) <= 0 )
                {
                  *(_BYTE *)(v45 + 146) |= 2u;
                  i = 1;
                  v61 = 1;
                  goto LABEL_174;
                }
              }
            }
            v45 = *(_QWORD *)v45;
            if ( !v45 )
              goto LABEL_173;
          }
        }
        v27 = v6;
      }
      goto LABEL_78;
    }
LABEL_81:
    v27 = v28;
    v28 = *(_QWORD *)v28;
    if ( v28 )
      continue;
    break;
  }
LABEL_82:
  i = v26;
  if ( v63 )
  {
    *v58 = 1;
    *a1 = 0;
    goto LABEL_87;
  }
  if ( !dword_4F077BC )
  {
    v61 = 0;
    *a1 = 0;
    goto LABEL_87;
  }
LABEL_176:
  if ( (unsigned __int64)(qword_4F077A8 - 40000LL) > 0x18F && !(_DWORD)qword_4F077B4 )
  {
    if ( i )
      goto LABEL_174;
    a5 = 0;
    v52 = 0;
    v7 = v6;
    a6 = 7;
    while ( 1 )
    {
      a3 = *(_QWORD **)(v7 + 120);
      if ( a3 )
      {
        v51 = 0;
        do
        {
          v50 = *((_DWORD *)a3 + 2);
          if ( v50 <= v51 )
            v50 = v51;
          else
            v51 = *((_DWORD *)a3 + 2);
          a3 = (_QWORD *)*a3;
        }
        while ( a3 );
      }
      else
      {
        v50 = 0;
        v51 = 0;
      }
      if ( v50 >= (int)a6 )
      {
        if ( !v52
          || (a3 = (_QWORD *)*(unsigned __int8 *)(v7 + 145), ((unsigned __int8)a3 & 2) != 0)
          || v51 != (_DWORD)a6 )
        {
          if ( v50 <= (int)a6 )
          {
            if ( v51 != (_DWORD)a6 || (v52 & 1) != 0 )
            {
              a5 = 0;
            }
            else
            {
              a3 = (_QWORD *)(*(_BYTE *)(v7 + 145) & 2);
              if ( (_BYTE)a3 )
                v52 = 0;
              else
                a5 = 0;
            }
          }
          goto LABEL_167;
        }
      }
      else
      {
        a3 = (_QWORD *)*(unsigned __int8 *)(v7 + 145);
      }
      LOBYTE(a3) = (unsigned __int8)a3 >> 1;
      a6 = (unsigned int)v51;
      a5 = v7;
      v52 = (unsigned __int8)a3 & 1;
LABEL_167:
      v7 = *(_QWORD *)v7;
      if ( !v7 )
      {
        if ( a5 && (qword_4F077A8 > 0x9C3Fu || *(_QWORD *)(a5 + 48)) )
        {
          v53 = v6;
          do
          {
            LOBYTE(v7) = a5 == v53;
            a3 = (_QWORD *)((unsigned int)v7 | *(_BYTE *)(v53 + 146) & 0xFE);
            *(_BYTE *)(v53 + 146) = (a5 == v53) | *(_BYTE *)(v53 + 146) & 0xFE;
            v53 = *(_QWORD *)v53;
          }
          while ( v53 );
LABEL_173:
          i = 1;
        }
        else
        {
          i = 0;
          v61 = 1;
        }
LABEL_174:
        *a1 = 0;
        goto LABEL_87;
      }
    }
  }
  if ( i )
    goto LABEL_174;
  v61 = 1;
  *a1 = 0;
LABEL_87:
  while ( 2 )
  {
    v30 = 0;
    while ( 2 )
    {
      v32 = v6;
      v6 = *(_QWORD *)v6;
      *(_QWORD *)v32 = 0;
      if ( *v58 )
      {
LABEL_97:
        sub_725130(*(__int64 **)(v32 + 40));
        sub_82D8A0(*(_QWORD **)(v32 + 120));
        a3 = qword_4D03C68;
        *(_QWORD *)v32 = qword_4D03C68;
        qword_4D03C68 = (_QWORD *)v32;
        goto LABEL_93;
      }
      if ( v61 )
      {
        if ( (*(_BYTE *)(v32 + 146) & 2) != 0 )
          goto LABEL_91;
        v31 = *(_QWORD *)(v32 + 8);
        if ( v31 )
        {
          if ( (*(_BYTE *)(v31 + 82) & 4) != 0 )
            goto LABEL_91;
        }
        goto LABEL_97;
      }
      if ( (*(_BYTE *)(v32 + 146) & 1) == 0 )
        goto LABEL_97;
LABEL_91:
      if ( v30 )
      {
        *v30 = v32;
        v30 = (_QWORD *)v32;
      }
      else
      {
        v30 = (_QWORD *)v32;
        *a1 = v32;
      }
LABEL_93:
      if ( v6 )
        continue;
      break;
    }
    v6 = *a1;
    if ( i )
      break;
LABEL_98:
    a6 = dword_4F077BC;
    if ( dword_4F077BC )
    {
      v61 = qword_4F077B4;
      if ( !(_DWORD)qword_4F077B4 )
      {
        if ( qword_4F077A8 )
        {
          v33 = *(_QWORD *)v6;
          if ( ((*(_BYTE *)(v6 + 145) ^ *(_BYTE *)(*(_QWORD *)v6 + 145LL)) & 2) != 0 )
          {
            v7 = *(_QWORD *)(v6 + 8);
            if ( v7 )
            {
              a3 = *(_QWORD **)(v33 + 8);
              if ( a3 )
              {
                if ( !*(_QWORD *)v33 )
                {
                  if ( (unsigned __int8)(*(_BYTE *)(v7 + 80) - 10) <= 1u )
                  {
                    v35 = *(_QWORD *)(v7 + 96);
                    if ( v35 )
                      v7 = *(_QWORD *)(v35 + 32);
                  }
                  if ( (unsigned __int8)(*((_BYTE *)a3 + 80) - 10) <= 1u )
                  {
                    v34 = a3[12];
                    if ( v34 )
                      a3 = *(_QWORD **)(v34 + 32);
                  }
                  if ( (_QWORD *)v7 == a3 )
                  {
                    if ( (*(_BYTE *)(v6 + 145) & 2) != 0 )
                    {
                      *(_BYTE *)(v6 + 146) &= ~1u;
                      *(_BYTE *)(v33 + 146) |= 1u;
                    }
                    else
                    {
                      *(_BYTE *)(v6 + 146) |= 1u;
                      *(_BYTE *)(v33 + 146) &= ~1u;
                    }
                    i = 1;
                    *a1 = 0;
                    continue;
                  }
                }
              }
            }
          }
LABEL_26:
          *a4 = 1;
          return;
        }
      }
    }
    break;
  }
  if ( *v58 )
    goto LABEL_26;
  if ( !v6 )
    return;
  if ( *(_QWORD *)v6 )
    goto LABEL_26;
LABEL_128:
  v40 = *(_QWORD *)(v6 + 8);
  if ( v40 && (*(_BYTE *)(v40 + 82) & 4) != 0 )
    goto LABEL_26;
  v14 = *(_BYTE *)(v6 + 32) == 0;
  *(_QWORD *)(v6 + 16) = v40;
  if ( !v14 )
    sub_82B390(v6, a2, a3, (_QWORD *)v7, a5, (_QWORD *)a6);
}
