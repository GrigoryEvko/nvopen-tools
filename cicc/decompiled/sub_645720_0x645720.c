// Function: sub_645720
// Address: 0x645720
//
__int64 __fastcall sub_645720(unsigned int a1, __int64 a2, __int64 a3, __int64 a4)
{
  __int64 v4; // rax
  _QWORD *v5; // rbx
  int v6; // r12d
  int v7; // r13d
  int v8; // r15d
  int v9; // eax
  __int64 v10; // r14
  __int64 v11; // rdi
  _QWORD *v12; // rbx
  _QWORD *v13; // rcx
  __int64 v14; // r12
  char v15; // dl
  __int64 v16; // rax
  __int64 v17; // r12
  char v18; // dl
  __int64 v19; // rax
  __int64 v20; // rdi
  __int64 v22; // rsi
  __int64 v23; // rcx
  __int64 v24; // r8
  int v25; // eax
  int v26; // eax
  __int64 v27; // r8
  bool v28; // si
  __int64 v29; // rcx
  __int64 j; // r12
  char i; // al
  int v32; // eax
  char v33; // r9
  __int64 v34; // rax
  bool v35; // r12
  __int64 v36; // rdi
  int v37; // eax
  _QWORD *v38; // rcx
  int v39; // eax
  int v40; // edi
  int v41; // eax
  __int64 v42; // rdx
  __int64 v43; // r8
  __int64 v44; // rax
  __int64 v45; // rdi
  __int64 v46; // rax
  __int64 v47; // rax
  int v48; // eax
  _QWORD *v49; // r14
  __int64 v50; // rdx
  char k; // al
  __int64 v52; // rax
  __int64 v53; // rax
  __int64 v54; // rdi
  char *v55; // rdx
  unsigned int v56; // [rsp+10h] [rbp-70h]
  char v58; // [rsp+18h] [rbp-68h]
  __int64 v59; // [rsp+20h] [rbp-60h]
  _BYTE *v60; // [rsp+28h] [rbp-58h]
  _QWORD **v61; // [rsp+28h] [rbp-58h]
  int v62; // [rsp+28h] [rbp-58h]
  _QWORD *v63; // [rsp+28h] [rbp-58h]
  _QWORD *v64; // [rsp+28h] [rbp-58h]
  __int64 v65; // [rsp+28h] [rbp-58h]
  unsigned __int8 v68; // [rsp+44h] [rbp-3Ch]
  char v69; // [rsp+48h] [rbp-38h]
  __int64 v70; // [rsp+48h] [rbp-38h]

  v4 = *(_QWORD *)(a2 + 168);
  v68 = a1;
  v5 = *(_QWORD **)v4;
  v60 = (_BYTE *)v4;
  v6 = *(_QWORD *)(v4 + 40) != 0;
  v59 = *(_QWORD *)(v4 + 40);
  v69 = a1 & 0xFD;
  if ( *(_QWORD *)v4 )
  {
    v7 = 0;
    v8 = 0;
    do
    {
      while ( 1 )
      {
        v10 = v5[1];
        ++v6;
        if ( (unsigned int)sub_8D32E0(v10) )
          v10 = sub_8D46C0(v10);
        if ( !(unsigned int)sub_8D3A70(v10) && (!unk_4D047E4 || !(unsigned int)sub_8D2870(v10)) )
          break;
        v5 = (_QWORD *)*v5;
        v8 = 1;
        if ( !v5 )
          goto LABEL_12;
      }
      v9 = sub_8D3D40(v10);
      v5 = (_QWORD *)*v5;
      if ( v9 )
        v7 = 1;
    }
    while ( v5 );
LABEL_12:
    if ( v69 != 1 && (((_BYTE)a1 - 2) & 0xFD) != 0 )
    {
      if ( (_BYTE)a1 != 42 )
        goto LABEL_15;
LABEL_37:
      LODWORD(v12) = 0;
      return (unsigned int)v12;
    }
    goto LABEL_20;
  }
  if ( v69 == 1 || (((_BYTE)a1 - 2) & 0xFD) == 0 || (_BYTE)a1 == 42 )
  {
    if ( !v59 )
    {
      if ( (_BYTE)a1 == 42 )
        goto LABEL_37;
      v40 = -((*(_BYTE *)(v4 + 16) & 1) == 0);
      LOBYTE(v40) = v40 & 0x2A;
      v11 = (unsigned int)(v40 + 559);
      if ( a4 )
      {
        v7 = 0;
        v8 = 0;
LABEL_18:
        LODWORD(v12) = 1;
        sub_6851C0(v11, a4);
        goto LABEL_26;
      }
      v7 = 0;
      v8 = 0;
      LODWORD(v12) = 1;
LABEL_26:
      if ( v69 == 1 )
        goto LABEL_27;
      goto LABEL_143;
    }
    v7 = 0;
    v8 = 0;
LABEL_20:
    if ( (_BYTE)a1 == 42 )
      goto LABEL_37;
    v13 = *(_QWORD **)v60;
    v14 = *(_QWORD *)(*(_QWORD *)v60 + 8LL);
    v15 = *(_BYTE *)(v14 + 140);
    if ( v15 == 12 )
    {
      v16 = *(_QWORD *)(*(_QWORD *)v60 + 8LL);
      do
      {
        v16 = *(_QWORD *)(v16 + 160);
        v15 = *(_BYTE *)(v16 + 140);
      }
      while ( v15 == 12 );
    }
    if ( !v15 )
    {
      LODWORD(v12) = 0;
      goto LABEL_26;
    }
    if ( v69 == 1 )
    {
      v45 = *(_QWORD *)(*(_QWORD *)v60 + 8LL);
      v65 = *(_QWORD *)v60;
      if ( !(unsigned int)sub_8D2780(v14) )
        goto LABEL_102;
      while ( *(_BYTE *)(v14 + 140) == 12 )
        v14 = *(_QWORD *)(v14 + 160);
      if ( *(_BYTE *)(v14 + 160) != unk_4F06A51 )
      {
LABEL_102:
        LODWORD(v12) = 1;
        v46 = sub_72C930(v45);
        v11 = 351;
        *(_QWORD *)(v65 + 8) = v46;
        if ( !a4 )
          goto LABEL_27;
        goto LABEL_18;
      }
    }
    else
    {
      if ( !unk_4D04814
        || (_BYTE)a1 != 2
        || !a3
        || !*v13
        || (v61 = *(_QWORD ***)v60, v22 = sub_72D2E0(a3, 0), v25 = sub_8D97D0(v14, v22, 0, v23, v24), v13 = v61, !v25)
        || (v26 = sub_8D3D00((*v61)[1]), v13 = v61, !v26) )
      {
        v36 = v14;
        v63 = v13;
        v37 = sub_8D4C80(v14);
        v38 = v63;
        v33 = (v68 - 2) & 0xFD;
        if ( v37 )
        {
          if ( unk_4D04814 && a3 )
          {
            LODWORD(v12) = 0;
            if ( !*v63 )
              goto LABEL_92;
            v39 = sub_8D3D00(*(_QWORD *)(*v63 + 8LL));
            v33 = (v68 - 2) & 0xFD;
            v38 = v63;
            if ( v39 )
            {
              if ( a4 )
              {
                sub_72D2E0(a3, 0);
                LODWORD(v12) = 1;
                sub_685360(3034, a4);
                v33 = (v68 - 2) & 0xFD;
                goto LABEL_92;
              }
            }
          }
        }
        else
        {
          if ( !HIDWORD(qword_4D0495C) || (v36 = v14, v41 = sub_8D2710(v14), v38 = v63, v33 = (v68 - 2) & 0xFD, !v41) )
          {
            v58 = v33;
            LODWORD(v12) = 1;
            v64 = v38;
            v44 = sub_72C930(v36);
            v33 = v58;
            v64[1] = v44;
            if ( a4 )
            {
              sub_684AA0(8, 354, a4);
              v33 = v58;
            }
LABEL_92:
            if ( v33 )
              goto LABEL_93;
LABEL_27:
            v17 = *(_QWORD *)(a2 + 160);
            v18 = *(_BYTE *)(v17 + 140);
            if ( v18 == 12 )
            {
              v19 = *(_QWORD *)(a2 + 160);
              do
              {
                v19 = *(_QWORD *)(v19 + 160);
                v18 = *(_BYTE *)(v19 + 140);
              }
              while ( v18 == 12 );
            }
            if ( v18 )
            {
              v20 = *(_QWORD *)(a2 + 160);
              if ( v69 == 1 )
              {
                if ( !(unsigned int)sub_8D4C80(v20)
                  || (*(_BYTE *)(v17 + 140) & 0xFB) == 8 && (unsigned int)sub_8D4C10(v17, dword_4F077C4 != 2) )
                {
                  if ( a4 )
                  {
                    LODWORD(v12) = 1;
                    sub_6851C0(352, a4);
                    return (unsigned int)v12;
                  }
LABEL_81:
                  LODWORD(v12) = 1;
                  return (unsigned int)v12;
                }
              }
              else if ( !(unsigned int)sub_8D2600(v20)
                     || (*(_BYTE *)(v17 + 140) & 0xFB) == 8 && (unsigned int)sub_8D4C10(v17, dword_4F077C4 != 2) )
              {
                if ( a4 )
                  sub_6851C0(353, a4);
                goto LABEL_81;
              }
            }
            return (unsigned int)v12;
          }
          v52 = sub_72CBE0(v14, v68, v42, v63, v43, (unsigned __int8)(v68 - 2) & 0xFD);
          v53 = sub_72D2E0(v52, 0);
          v38 = v63;
          v33 = (v68 - 2) & 0xFD;
          v63[1] = v53;
          if ( a4 )
          {
            sub_684AA0(5, 354, a4);
            v38 = v63;
            v33 = (v68 - 2) & 0xFD;
          }
        }
        v12 = (_QWORD *)*v38;
        v28 = dword_4D048B8 == 0;
        if ( !*v38 )
          goto LABEL_107;
        v29 = 0;
LABEL_48:
        j = v12[1];
        for ( i = *(_BYTE *)(j + 140); i == 12; i = *(_BYTE *)(j + 140) )
          j = *(_QWORD *)(j + 160);
        if ( i )
        {
          v56 = v29;
          v48 = sub_8D2780(j);
          v29 = v56;
          if ( v48 )
          {
            if ( *(_BYTE *)(j + 160) == unk_4F06A51 )
            {
              if ( a3 || unk_4D04478 )
              {
                v12 = (_QWORD *)*v12;
                if ( !v12 )
                {
LABEL_143:
                  v33 = (v68 - 2) & 0xFD;
                  goto LABEL_92;
                }
                for ( j = v12[1]; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
                  ;
              }
              else if ( a4 && v28 )
              {
                sub_684AA0(5 - (unsigned int)(dword_4D048B8 == 0), 831, a4);
                v29 = v56;
              }
            }
          }
        }
        if ( unk_4D04818 )
        {
          if ( unk_4F06C60 == j || (v62 = v29, v32 = sub_8D97D0(j, unk_4F06C60, 0, v29, v27), LODWORD(v29) = v62, v32) )
          {
            if ( !*v12 )
            {
              LODWORD(v12) = 0;
              v33 = (v68 - 2) & 0xFD;
              goto LABEL_92;
            }
          }
        }
        v33 = (v68 - 2) & 0xFD;
        if ( a4 && (_DWORD)v29 )
        {
          LODWORD(v12) = 1;
          sub_6851C0(3035, a4);
          v33 = (v68 - 2) & 0xFD;
          goto LABEL_92;
        }
LABEL_107:
        LODWORD(v12) = 0;
        goto LABEL_92;
      }
      v12 = (_QWORD *)**v61;
      v28 = dword_4D048B8 == 0;
      if ( v12 )
      {
        v29 = 1;
        goto LABEL_48;
      }
    }
    LODWORD(v12) = 0;
    goto LABEL_27;
  }
  v8 = 0;
  v7 = 0;
LABEL_15:
  if ( (v60[16] & 1) != 0 )
  {
    v11 = 559;
    goto LABEL_17;
  }
  if ( (unsigned __int8)a1 > 0x2Fu )
  {
    if ( v6 == 1 )
    {
LABEL_111:
      v11 = 345;
LABEL_17:
      LODWORD(v12) = 1;
      if ( !a4 )
      {
LABEL_93:
        v35 = v59 == 0;
        goto LABEL_63;
      }
      goto LABEL_18;
    }
  }
  else
  {
    v34 = 0x820000006000LL;
    if ( _bittest64(&v34, a1) )
    {
      v11 = 344;
      if ( v6 > 1 )
        goto LABEL_17;
      if ( v6 )
      {
LABEL_62:
        v35 = v59 == 0;
        LODWORD(v12) = 0;
        goto LABEL_63;
      }
      goto LABEL_111;
    }
    if ( v6 == 1 )
    {
      if ( (unsigned __int8)a1 <= 0x26u )
      {
        v47 = 0x60000008E0LL;
        if ( _bittest64(&v47, (unsigned __int8)a1) )
          goto LABEL_62;
      }
      goto LABEL_111;
    }
  }
  if ( v6 != 2 )
  {
    v11 = (unsigned int)(v6 < 3) + 344;
    goto LABEL_17;
  }
  v35 = v59 == 0;
  LODWORD(v12) = 0;
  if ( (unsigned __int8)(a1 - 37) <= 1u )
  {
    v49 = *(_QWORD **)v60;
    if ( !v59 )
      v49 = (_QWORD *)*v49;
    v50 = v49[1];
    for ( k = *(_BYTE *)(v50 + 140); k == 12; k = *(_BYTE *)(v50 + 140) )
      v50 = *(_QWORD *)(v50 + 160);
    LODWORD(v12) = 0;
    if ( k )
    {
      v70 = v50;
      LODWORD(v12) = sub_8DBE70(v50);
      if ( (_DWORD)v12 )
      {
        LODWORD(v12) = 0;
      }
      else
      {
        v54 = v70;
        if ( !(unsigned int)sub_8D2780(v70) || *(_BYTE *)(v70 + 160) != 5 )
        {
          if ( a4 )
          {
            v55 = "++";
            v54 = 500;
            if ( v68 != 37 )
              v55 = "--";
            sub_6851A0(500, a4, v55);
          }
          LODWORD(v12) = 1;
          v49[1] = sub_72C930(v54);
        }
      }
    }
  }
LABEL_63:
  if ( v68 != 42 && v35 && !(v7 | v8) )
  {
    if ( a4 )
    {
      LODWORD(v12) = 1;
      sub_6851C0(unk_4D047E4 == 0 ? 346 : 898, a4);
      return (unsigned int)v12;
    }
    goto LABEL_81;
  }
  return (unsigned int)v12;
}
