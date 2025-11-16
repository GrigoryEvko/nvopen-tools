// Function: sub_875E10
// Address: 0x875e10
//
__int64 __fastcall sub_875E10(int a1, __int64 a2, FILE *a3, int a4, __int64 a5)
{
  __int64 result; // rax
  int v6; // ebx
  __int64 v7; // r14
  char v9; // r13
  char v11; // al
  __int64 v12; // rax
  __int64 v13; // rcx
  __int64 v14; // rdx
  __int64 v15; // rcx
  __int64 v16; // rax
  __int64 v17; // rax
  _BYTE *v18; // rdx
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // rdx
  __int64 v23; // rax
  __int64 v24; // rax
  char v25; // dl
  int v26; // edi
  __int64 v27; // rdi
  char v28; // al
  __int64 v29; // rax
  __int64 v30; // rax
  __int64 v31; // rdx
  __int64 v32; // rcx
  __int64 v33; // rdi
  char i; // si
  __int64 v35; // rax
  char v36; // dl
  __int16 v37; // dx
  __int64 v38; // rax
  char v39; // al
  int v40; // eax
  __int64 v41; // rdi
  __int64 v42; // rcx
  int v43; // eax
  __int64 v44; // rax
  __int64 v45; // [rsp+0h] [rbp-50h]
  __int64 v46; // [rsp+8h] [rbp-48h]
  __int64 v47; // [rsp+8h] [rbp-48h]
  __int64 v48; // [rsp+8h] [rbp-48h]
  __int64 v49; // [rsp+10h] [rbp-40h]
  __int64 v50; // [rsp+10h] [rbp-40h]
  __int64 v51; // [rsp+10h] [rbp-40h]
  __int64 v52; // [rsp+10h] [rbp-40h]
  __int64 v53; // [rsp+10h] [rbp-40h]
  __int64 v54; // [rsp+10h] [rbp-40h]
  __int64 v55; // [rsp+10h] [rbp-40h]

  result = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( (*(_BYTE *)(result + 12) & 4) != 0 )
    return result;
  v6 = a1;
  v7 = a2;
  v9 = *(_BYTE *)(a2 + 80);
  if ( !qword_4D04900 )
    goto LABEL_7;
  if ( v9 != 3 )
  {
    if ( (unsigned __int8)(v9 - 4) <= 1u )
    {
      v22 = *(_QWORD *)(a2 + 88);
      if ( (unsigned __int8)(*(_BYTE *)(v22 + 140) - 9) <= 2u
        && (*(_BYTE *)(a2 + 81) & 0x10) == 0
        && *(char *)(v22 + 177) < 0 )
      {
        a2 = *(_QWORD *)(*(_QWORD *)(a2 + 96) + 72LL);
        goto LABEL_6;
      }
    }
LABEL_5:
    LOWORD(a1) = v6;
    a2 = v7;
    goto LABEL_6;
  }
  if ( (*(_BYTE *)(a2 + 81) & 0x40) == 0 )
    goto LABEL_5;
  v27 = *(_QWORD *)(a2 + 88);
  v28 = *(_BYTE *)(v27 + 140);
  if ( v28 == 14 )
    goto LABEL_5;
  while ( v28 != 12 )
  {
    if ( (unsigned int)sub_8D3410(v27) )
    {
      v27 = sub_8D4050(v27);
    }
    else
    {
      if ( !(unsigned int)sub_8D3320(v27) )
        goto LABEL_5;
      v27 = sub_8D46C0(v27);
    }
LABEL_86:
    v28 = *(_BYTE *)(v27 + 140);
  }
  a2 = *(_QWORD *)v27;
  if ( !*(_QWORD *)v27 )
  {
    v27 = *(_QWORD *)(v27 + 160);
    goto LABEL_86;
  }
  LOWORD(a1) = v6 | 0x400;
LABEL_6:
  sub_8754F0(a1, (unsigned __int8 *)a2, (__int64)a3);
LABEL_7:
  if ( (*(_BYTE *)(v7 + 81) & 1) == 0 )
  {
    v11 = *(_BYTE *)(v7 + 80);
    if ( (unsigned __int8)(v11 - 10) <= 1u || v11 == 17 )
    {
      if ( (*(_BYTE *)(*(_QWORD *)(v7 + 88) + 195LL) & 3) != 1 )
        goto LABEL_8;
      v12 = *(_QWORD *)(v7 + 96);
    }
    else
    {
      if ( ((v11 - 7) & 0xFD) != 0 || (*(_BYTE *)(*(_QWORD *)(v7 + 88) + 170LL) & 0x90) != 0x10 )
        goto LABEL_8;
      v12 = sub_892240(v7, a2);
    }
    if ( v12 )
      *(_QWORD *)(v12 + 92) = *(_QWORD *)&a3->_flags;
  }
LABEL_8:
  *(_BYTE *)(v7 + 81) |= 1u;
  if ( !a5 )
    a5 = sub_87D520(v7);
  if ( !a4 )
    goto LABEL_20;
  if ( !a5 )
    goto LABEL_15;
  if ( v9 == 10 )
  {
    v29 = *(_QWORD *)(v7 + 88);
    if ( (*(_BYTE *)(v29 + 192) & 2) == 0 )
      goto LABEL_14;
    if ( (v6 & 0x20) == 0 )
      goto LABEL_21;
    goto LABEL_94;
  }
  if ( v9 == 9 && (v6 & 0x16000) == 0 )
  {
    if ( !dword_4F077BC )
    {
      v23 = *(_QWORD *)(v7 + 88);
      goto LABEL_67;
    }
    if ( (_DWORD)qword_4F077B4 || (v6 & 0x38) != 0 )
    {
      v23 = *(_QWORD *)(v7 + 88);
      v37 = *(_WORD *)(v23 + 176);
      LOBYTE(v37) = v37 & 1;
      if ( v37 == 1 )
      {
        v53 = *(_QWORD *)(v7 + 88);
        sub_5EB3F0((_QWORD *)v23);
        v23 = v53;
      }
LABEL_67:
      a2 = 1;
      v51 = v23;
      sub_8AD0D0(v7, 1, 1);
      *(_BYTE *)(v51 + 169) |= 0x10u;
    }
  }
LABEL_14:
  *(_BYTE *)(a5 + 88) |= 4u;
LABEL_15:
  if ( (v6 & 0x20) == 0 )
  {
LABEL_20:
    if ( ((v9 - 7) & 0xFD) != 0 )
      goto LABEL_21;
LABEL_36:
    v13 = 0;
    if ( dword_4F04C58 != -1 )
      v13 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216);
    v14 = *(_QWORD *)(v7 + 88);
    v49 = v13;
    if ( !v14 )
      goto LABEL_42;
    if ( (v6 & 0x12040) != 0 )
      goto LABEL_42;
    if ( (*(_BYTE *)(v14 + 156) & 7) != 0 )
      goto LABEL_42;
    v46 = *(_QWORD *)(v7 + 88);
    v45 = *(_QWORD *)(v14 + 40);
    if ( (unsigned int)sub_8D2FF0(*(_QWORD *)(v14 + 120), a2) || (unsigned int)sub_8D3030(*(_QWORD *)(v46 + 120)) )
      goto LABEL_42;
    v31 = v46;
    v32 = v49;
    v33 = *(_QWORD *)(v46 + 120);
    i = *(_BYTE *)(v33 + 140);
    if ( (i & 0xFB) == 8 )
    {
      if ( (sub_8D4C10(v33, dword_4F077C4 != 2) & 1) != 0 )
      {
LABEL_42:
        if ( v9 != 7 )
        {
          if ( v9 == 9 && (v6 & 0x2070) != 0 )
            *(_BYTE *)(v7 + 84) |= 0x80u;
          goto LABEL_21;
        }
        v15 = *(_QWORD *)(v7 + 88);
        if ( (v6 & 0x12068) == 0 )
          goto LABEL_50;
        if ( (*(_BYTE *)(v15 + 169) & 0x10) != 0 )
        {
          if ( (*(_DWORD *)(v15 + 168) & 0x80008000) != 0 )
            *(_BYTE *)(v15 + 171) |= 0x40u;
          goto LABEL_50;
        }
        if ( (v6 & 8) == 0
          || unk_4D048C4
          || *(char *)(v7 + 84) < 0
          || (*(_BYTE *)(v15 + 89) & 1) == 0
          || (*(_BYTE *)(v15 + 170) & 1) != 0 )
        {
          goto LABEL_108;
        }
        if ( dword_4F077C4 == 2 )
        {
          v47 = *(_QWORD *)(v7 + 88);
          v55 = *(_QWORD *)(v15 + 120);
          v40 = sub_8D3410(v55);
          v41 = v55;
          v42 = v47;
          if ( v40 )
          {
            v44 = sub_8D40F0(v55);
            v42 = v47;
            v41 = v44;
          }
          v48 = v42;
          v43 = sub_8D3A70(v41);
          v15 = v48;
          if ( v43 )
          {
            for ( ; *(_BYTE *)(v41 + 140) == 12; v41 = *(_QWORD *)(v41 + 160) )
              ;
            if ( *(char *)(*(_QWORD *)(*(_QWORD *)v41 + 96LL) + 179LL) >= 0 )
              goto LABEL_108;
          }
        }
        v30 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
        if ( qword_4F04C68[0] == v30 )
          goto LABEL_108;
        while ( 1 )
        {
          if ( *(_BYTE *)(v30 + 4) == 17 )
          {
            if ( (*(_BYTE *)(*(_QWORD *)(v30 + 216) + 206LL) & 2) == 0 || *(_DWORD *)v30 == *(_DWORD *)(v7 + 40) )
              goto LABEL_145;
          }
          else
          {
            if ( *(_DWORD *)v30 == *(_DWORD *)(v7 + 40) )
            {
              v30 = qword_4F04C68[0] + 776LL * dword_4F04C58;
LABEL_145:
              if ( *(_DWORD *)(v30 + 416) <= *(_DWORD *)(v7 + 44) && (*(_BYTE *)(v15 + 169) & 8) != 0 )
              {
                v54 = v15;
                sub_685490(0x225u, a3, v7);
                v15 = v54;
              }
LABEL_108:
              *(_BYTE *)(v15 + 169) |= 0x10u;
LABEL_50:
              if ( (v6 & 0x2070) != 0 )
              {
                v52 = v15;
                sub_8756B0(v7);
                v15 = v52;
                if ( dword_4D048B8 )
                {
                  if ( (*(_BYTE *)(v52 + 89) & 1) != 0 && (*(_BYTE *)(v52 + 170) & 8) == 0 )
                  {
                    v24 = qword_4F04C68[0] + 776LL * (int)dword_4F04C5C;
                    v25 = *(_BYTE *)(v24 + 7);
                    if ( v25 < 0 )
                    {
                      v26 = *(_DWORD *)(v7 + 40);
                      if ( v26 != *(_DWORD *)v24 )
                      {
                        while ( (v25 & 0x40) == 0 )
                        {
                          if ( *(_DWORD *)(v24 - 776) == v26 )
                            goto LABEL_51;
                          v25 = *(_BYTE *)(v24 - 769);
                          v24 -= 776;
                        }
                        *(_BYTE *)(v52 + 170) |= 8u;
                      }
                    }
                  }
                }
              }
LABEL_51:
              if ( (v6 & 0x13078) != 0
                && ((*(_BYTE *)(v15 + 172) & 0x20) != 0 || (*(_BYTE *)(v15 + 170) & 0xD0) == 0x10) )
              {
                v50 = v15;
                sub_8AD0D0(v7, 1, 1);
                v16 = *(_QWORD *)(v50 + 216);
                if ( v16 )
                {
                  v17 = *(_QWORD *)(v16 + 16);
                  v18 = *(_BYTE **)(v17 + 192);
                  *(_BYTE *)(*(_QWORD *)v17 + 81LL) |= 1u;
                  if ( a4 )
                  {
                    *(_BYTE *)(v17 + 88) |= 4u;
                    if ( v18 )
                    {
                      *(_BYTE *)(*(_QWORD *)v18 + 81LL) |= 1u;
                      v18[88] |= 4u;
                    }
                  }
                  else if ( v18 )
                  {
                    *(_BYTE *)(*(_QWORD *)v18 + 81LL) |= 1u;
                  }
                }
              }
              goto LABEL_21;
            }
            if ( (*(_BYTE *)(v30 + 5) & 0x20) != 0 )
              goto LABEL_108;
          }
          v30 -= 776;
          if ( qword_4F04C68[0] == v30 )
            goto LABEL_108;
        }
      }
      v31 = v46;
      v32 = v49;
      v38 = *(_QWORD *)(v46 + 120);
      for ( i = *(_BYTE *)(v38 + 140); i == 12; i = *(_BYTE *)(v38 + 140) )
        v38 = *(_QWORD *)(v38 + 160);
    }
    if ( i != 14 && (*(_BYTE *)(v31 + 174) & 4) == 0 )
    {
      if ( v45 )
      {
        if ( *(_BYTE *)(v7 + 80) != 7 || (v39 = *(_BYTE *)(v45 + 28), v39 == 3) || !v39 )
        {
          if ( v32
            && (*(_BYTE *)(v32 + 198) & 0x10) != 0
            && (*(_BYTE *)(v32 + 193) & 0x10) == 0
            && ((*(_QWORD *)(v32 + 200) & 0x8000001000000LL) != 0x8000000000000LL || (*(_BYTE *)(v32 + 192) & 2) != 0) )
          {
            if ( (v6 & 8) != 0 )
              sub_685490(0xDDFu, a3, v7);
            if ( (v6 & 0x10) != 0 )
              sub_685490(0xDE0u, a3, v7);
            if ( (v6 & 0x20) != 0 )
              sub_685490(0xDE1u, a3, v7);
          }
        }
      }
    }
    goto LABEL_42;
  }
  if ( v9 == 7 || v9 == 9 )
  {
    sub_72A420(*(__int64 **)(v7 + 88));
    goto LABEL_36;
  }
  if ( (unsigned __int8)(v9 - 10) <= 1u )
  {
    v29 = *(_QWORD *)(v7 + 88);
LABEL_94:
    *(_BYTE *)(v29 + 192) |= 1u;
    goto LABEL_20;
  }
  if ( v9 != 8 )
    goto LABEL_20;
  v35 = *(_QWORD *)(v7 + 96);
  if ( v35 )
  {
    while ( 1 )
    {
      v36 = *(_BYTE *)(v35 + 80);
      if ( v36 == 7 )
        break;
      if ( v36 == 8 )
      {
        v35 = *(_QWORD *)(v35 + 96);
        if ( v35 )
          continue;
      }
      goto LABEL_21;
    }
    sub_72A420(*(__int64 **)(v35 + 88));
  }
LABEL_21:
  if ( (HIDWORD(qword_4F077B4) || dword_4F077C4 == 2 && unk_4F07778 > 201401) && a5 && (unsigned __int8)(v9 - 3) > 3u )
    sub_875AD0(a5, a3);
  result = (unsigned int)*(unsigned __int8 *)(v7 + 80) - 10;
  if ( (unsigned __int8)(*(_BYTE *)(v7 + 80) - 10) <= 1u )
  {
    result = sub_875C60(v7, 0, a3);
    if ( (*(_BYTE *)(v7 + 104) & 2) != 0 )
      return sub_894C00(v7, 0, v19, v20, v21);
  }
  return result;
}
