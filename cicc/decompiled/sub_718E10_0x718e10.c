// Function: sub_718E10
// Address: 0x718e10
//
__int64 __fastcall sub_718E10(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v7; // rbx
  char v8; // al
  __int64 v9; // r12
  char v11; // r14
  __int64 v12; // r9
  __int64 i; // r13
  __int64 v14; // r15
  unsigned __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rcx
  unsigned __int64 v18; // rax
  unsigned __int64 j; // r14
  __int64 v20; // r12
  __int64 v21; // r8
  int v22; // eax
  char v23; // dl
  unsigned __int64 v24; // rax
  char v25; // al
  int v26; // eax
  __int64 v27; // rax
  int v28; // eax
  char v29; // al
  __int64 v30; // rsi
  __int64 v31; // rax
  __int64 v32; // r11
  __int64 v33; // r10
  __int64 v34; // rsi
  unsigned __int64 v35; // rtt
  __int64 v36; // rax
  unsigned __int64 v37; // r14
  int v38; // eax
  __int64 v39; // rax
  __int64 v40; // r8
  __int64 v41; // r14
  __int64 v42; // rax
  __int64 v43; // rsi
  __int64 k; // rax
  __int64 v45; // rax
  __int64 v46; // [rsp+0h] [rbp-60h]
  _BOOL4 v47; // [rsp+Ch] [rbp-54h]
  __int64 v48; // [rsp+10h] [rbp-50h]
  __int64 v49; // [rsp+10h] [rbp-50h]
  __int64 v50; // [rsp+10h] [rbp-50h]
  __int64 v51; // [rsp+10h] [rbp-50h]
  __int64 v52; // [rsp+18h] [rbp-48h]
  __int64 v53; // [rsp+18h] [rbp-48h]
  __int64 v54; // [rsp+18h] [rbp-48h]
  __int64 v55; // [rsp+18h] [rbp-48h]
  __int64 v56; // [rsp+18h] [rbp-48h]
  __int64 v57; // [rsp+18h] [rbp-48h]
  __int64 v58; // [rsp+18h] [rbp-48h]
  __int64 v59; // [rsp+18h] [rbp-48h]
  __int64 v60; // [rsp+20h] [rbp-40h] BYREF
  __int64 v61[7]; // [rsp+28h] [rbp-38h] BYREF

  v7 = a2;
  v60 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v8 = *(_BYTE *)(a1 + 173);
  if ( !v8 )
  {
LABEL_2:
    if ( v7 )
    {
      v9 = v7;
      sub_72C970(v7);
    }
    else
    {
      v9 = sub_72C9A0();
    }
    goto LABEL_4;
  }
  if ( v8 != 6 )
    goto LABEL_28;
  v11 = *(_BYTE *)(a1 + 176);
  if ( (unsigned __int8)(v11 - 1) > 2u )
    goto LABEL_28;
  for ( i = sub_8D46C0(*(_QWORD *)(a1 + 128)); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  v14 = *(_QWORD *)(a1 + 184);
  v15 = v11 == 1 ? *(_QWORD *)(v14 + 120) : *(_QWORD *)(v14 + 128);
  v16 = *(_QWORD *)(a1 + 192);
  if ( v16 < 0 )
    goto LABEL_28;
  v17 = *(unsigned __int8 *)(v15 + 140);
  v18 = v15;
  if ( (_BYTE)v17 == 12 )
  {
    do
      v18 = *(_QWORD *)(v18 + 160);
    while ( *(_BYTE *)(v18 + 140) == 12 );
  }
  if ( (unsigned __int64)v16 >= *(_QWORD *)(v18 + 128) )
    goto LABEL_28;
  if ( v11 == 1 )
  {
    v15 = *(_QWORD *)(a1 + 184);
    v14 = sub_6EA7C0(v15);
  }
  else if ( v11 == 3 )
  {
    if ( (v17 & 0xFB) != 8 )
      goto LABEL_28;
    a2 = dword_4F077C4 != 2;
    if ( (sub_8D4C10(v15, a2) & 1) == 0 )
      goto LABEL_28;
    v14 = *(_QWORD *)(a1 + 184);
  }
  if ( !v14 )
    goto LABEL_28;
  if ( !*(_BYTE *)(v14 + 173) )
    goto LABEL_2;
  for ( j = *(_QWORD *)(v14 + 128); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
    ;
  v20 = *(_QWORD *)(a1 + 192);
  v21 = 0;
LABEL_22:
  while ( 2 )
  {
    while ( 2 )
    {
      if ( v20 == v21 )
      {
        v52 = v21;
        if ( i == j )
        {
          if ( !v14 )
            goto LABEL_26;
          goto LABEL_35;
        }
        a2 = j;
        v15 = i;
        v22 = sub_8D97D0(i, j, 0, v17, v21);
        v21 = v52;
        if ( v22 )
        {
          if ( !v14 )
            goto LABEL_26;
LABEL_34:
          a2 = j;
          v53 = v21;
          if ( (unsigned int)sub_8D97D0(i, j, 0, v17, v21) )
          {
LABEL_35:
            if ( v7 )
            {
              v9 = v7;
              sub_740190(v14, v7, 4352);
            }
            else
            {
              v9 = sub_740190(v14, 0, 4096);
            }
            goto LABEL_4;
          }
          v15 = *(_QWORD *)(v14 + 128);
          v36 = sub_8D4050(v15);
          v21 = v53;
          v37 = v36;
          if ( v36 == i || (a2 = v36, v15 = i, v38 = sub_8D97D0(i, v36, 32, v17, v53), v21 = v53, v38) )
          {
LABEL_97:
            if ( v20 >= v21 + *(_QWORD *)(v14 + 176) )
              goto LABEL_26;
            a2 = sub_722AB0(*(_QWORD *)(v14 + 184) + v20 - v21, *(unsigned int *)(i + 128));
            sub_72BAF0(v60, a2, *(unsigned __int8 *)(i + 160));
            v15 = v37;
            if ( (unsigned int)sub_8D2A50(v37) && unk_4F06B98 )
            {
              v15 = v60 + 176;
              a2 = dword_4F06BA0;
              sub_6215A0((__int16 *)(v60 + 176), dword_4F06BA0);
            }
            v14 = v60;
            if ( !v60 )
              goto LABEL_26;
            goto LABEL_35;
          }
          if ( (unsigned int)sub_8D2A50(v37) )
          {
            v15 = i;
            if ( (unsigned int)sub_8D29E0(i) )
            {
              v21 = v53;
              goto LABEL_97;
            }
          }
LABEL_28:
          v9 = 0;
          goto LABEL_4;
        }
        a2 = *(unsigned __int8 *)(v14 + 173);
        if ( (*(_BYTE *)(v14 + 173) & 0xF7) != 2 )
          goto LABEL_28;
      }
      else
      {
        a2 = *(unsigned __int8 *)(v14 + 173);
      }
      if ( (_BYTE)a2 == 2 )
      {
        if ( v21 + *(_QWORD *)(j + 128) <= v20 || !v14 )
          goto LABEL_26;
        if ( i == j )
          goto LABEL_35;
        goto LABEL_34;
      }
      v23 = *(_BYTE *)(j + 140);
      if ( v23 == 12 )
      {
        v24 = j;
        do
        {
          v24 = *(_QWORD *)(v24 + 160);
          v23 = *(_BYTE *)(v24 + 140);
        }
        while ( v23 == 12 );
      }
      if ( !v23 || (_BYTE)a2 != 10 )
        goto LABEL_28;
      v25 = *(_BYTE *)(v14 + 170);
      v15 = j;
      v54 = v21;
      v14 = *(_QWORD *)(v14 + 176);
      v47 = (v25 & 2) != 0;
      v26 = sub_8D3410(j);
      v21 = v54;
      if ( !v26 )
      {
        v12 = **(_QWORD **)(j + 168);
        if ( v12 )
        {
          while ( v14 )
          {
            v29 = *(_BYTE *)(v12 + 96);
            if ( (v29 & 1) != 0 )
            {
              v15 = *(_QWORD *)(v12 + 40);
              v30 = *(_QWORD *)(v15 + 168);
              v16 = *(_QWORD *)(v30 + 208);
              if ( v16 )
              {
                if ( (*(_BYTE *)(v30 + 109) & 0x10) == 0 )
                  v16 = *(_QWORD *)(v12 + 40);
              }
              else
              {
                v16 = *(_QWORD *)(v12 + 40);
              }
              a2 = 0;
              if ( (v29 & 0x20) == 0 )
                a2 = *(_QWORD *)(*(_QWORD *)(v16 + 168) + 32LL);
              v27 = v21 + *(_QWORD *)(v12 + 104);
              if ( v27 == v20 || v27 < v20 && a2 + v27 > v20 )
              {
                if ( a2 )
                  break;
                if ( v15 == i )
                  break;
                a2 = i;
                v48 = v21;
                v55 = v12;
                v28 = sub_8D97D0(v15, i, 0, v17, v21);
                v12 = v55;
                v21 = v48;
                if ( v28 )
                  break;
              }
              v14 = *(_QWORD *)(v14 + 120);
            }
            v12 = *(_QWORD *)v12;
            if ( !v12 )
              goto LABEL_107;
          }
          j = *(_QWORD *)(v12 + 40);
          v21 += *(_QWORD *)(v12 + 104);
        }
        else
        {
LABEL_107:
          if ( v14 )
          {
            if ( *(_BYTE *)(v14 + 173) == 13 && *(_BYTE *)(j + 140) == 11 )
            {
              for ( j = *(_QWORD *)(*(_QWORD *)(v14 + 184) + 120LL); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
                ;
              v14 = *(_QWORD *)(v14 + 120);
              goto LABEL_104;
            }
            v15 = *(_QWORD *)(j + 160);
            a2 = 7;
            v49 = v21;
            v56 = v12;
            v39 = sub_72FD90(v15, 7);
            v12 = v56;
            v40 = v49;
            v41 = v39;
          }
          else
          {
            v15 = *(_QWORD *)(j + 160);
            a2 = 7;
            v51 = v21;
            v59 = v12;
            v45 = sub_72FD90(v15, 7);
            v12 = v59;
            v40 = v51;
            v41 = v45;
            if ( !v45 )
              goto LABEL_145;
          }
          v16 = 0;
          do
          {
            if ( v14 && *(_BYTE *)(v14 + 173) == 13 )
            {
              v41 = *(_QWORD *)(v14 + 184);
              v14 = *(_QWORD *)(v14 + 120);
            }
            if ( !v41 )
              break;
            v43 = v40 + *(_QWORD *)(v41 + 128);
            if ( *(_BYTE *)(v41 + 137) || v20 < v43 )
              goto LABEL_112;
            for ( k = *(_QWORD *)(v41 + 120); *(_BYTE *)(k + 140) == 12; k = *(_QWORD *)(k + 160) )
              ;
            a2 = *(_QWORD *)(k + 128) + v43;
            if ( a2 <= v20 )
            {
LABEL_112:
              v15 = *(_QWORD *)(v41 + 112);
              a2 = 7;
              v46 = v16;
              v50 = v40;
              v57 = v12;
              v42 = sub_72FD90(v15, 7);
              v12 = v57;
              v40 = v50;
              v16 = v46;
              v41 = v42;
              if ( v14 )
              {
                v14 = *(_QWORD *)(v14 + 120);
                v42 |= v14;
              }
            }
            else
            {
              if ( !v47 )
              {
                v16 = v41;
                v12 = v14;
                goto LABEL_126;
              }
              v15 = *(_QWORD *)(v41 + 112);
              a2 = 7;
              v58 = v40;
              v42 = sub_72FD90(v15, 7);
              v40 = v58;
              if ( v14 )
              {
                a2 = *(_QWORD *)(v14 + 120);
                v16 = v41;
                v12 = v14;
                v41 = v42;
                v14 = a2;
                v42 |= a2;
              }
              else
              {
                v16 = v41;
                v12 = 0;
                v41 = v42;
              }
            }
          }
          while ( v42 );
          if ( !v16 )
LABEL_145:
            BUG();
LABEL_126:
          for ( j = *(_QWORD *)(v16 + 120); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          v21 = *(_QWORD *)(v16 + 128) + v40;
          v14 = v12;
        }
LABEL_104:
        if ( v14 )
          continue;
        goto LABEL_26;
      }
      break;
    }
    do
      j = *(_QWORD *)(j + 160);
    while ( *(_BYTE *)(j + 140) == 12 );
    if ( !v14 )
      break;
    v31 = v54;
    v32 = 0;
    v33 = 0;
    do
    {
      while ( 1 )
      {
        v15 = *(unsigned __int8 *)(v14 + 173);
        if ( (_BYTE)v15 == 13 )
        {
          v34 = *(_QWORD *)(j + 128) * *(_QWORD *)(v14 + 184);
          v14 = *(_QWORD *)(v14 + 120);
          a2 = v54 + v34;
          v15 = *(unsigned __int8 *)(v14 + 173);
          v16 = a2;
        }
        else
        {
          a2 = v31;
          v16 = v31;
        }
        if ( (_BYTE)v15 != 11 )
          break;
        v12 = *(_QWORD *)(v14 + 176);
        for ( j = *(_QWORD *)(v12 + 128); *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        v15 = *(_QWORD *)(j + 128);
        v31 = a2 + v15 * *(_QWORD *)(v14 + 184);
        if ( v16 > v20 || v20 >= v31 )
          goto LABEL_70;
        v35 = v20 - v16;
        v16 = (v20 - v16) % v15;
        v31 = a2 + v15 * (v35 / v15);
        if ( !v47 )
        {
          v14 = *(_QWORD *)(v14 + 176);
          v21 = v31;
          goto LABEL_22;
        }
        v32 = v31;
        v33 = *(_QWORD *)(v14 + 176);
        v14 = v33;
      }
      v31 = a2 + *(_QWORD *)(j + 128);
      if ( v16 <= v20 && v20 < v31 )
      {
        v17 = v47;
        if ( !v47 )
        {
          v21 = v16;
          goto LABEL_22;
        }
        v32 = v16;
        v33 = v14;
      }
LABEL_70:
      v14 = *(_QWORD *)(v14 + 120);
    }
    while ( v14 );
    if ( v33 )
    {
      v14 = v33;
      v21 = v32;
      continue;
    }
    break;
  }
LABEL_26:
  if ( v7 )
  {
    v9 = v7;
    if ( !(unsigned int)sub_72FDF0(i, v7) )
      goto LABEL_28;
  }
  else
  {
    v61[0] = sub_724DC0(v15, a2, v16, v17, v21, v12);
    if ( (unsigned int)sub_72FDF0(i, v61[0]) )
      v7 = sub_7401F0(v61[0]);
    v9 = v7;
    sub_724E30(v61);
  }
LABEL_4:
  sub_724E30(&v60);
  return v9;
}
