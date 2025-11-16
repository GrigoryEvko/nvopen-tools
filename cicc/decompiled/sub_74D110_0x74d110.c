// Function: sub_74D110
// Address: 0x74d110
//
void __fastcall sub_74D110(__int64 a1, int a2, int a3, __int64 a4)
{
  int v4; // ebx
  char v5; // al
  __int64 v8; // r15
  __int64 v9; // r14
  __int64 i; // rax
  unsigned int v11; // eax
  unsigned int (__fastcall *v12)(__int64, __int64 *); // rax
  __int64 *v13; // rax
  __int64 v14; // rdi
  __int64 v15; // rax
  char v16; // dl
  __int64 v17; // rax
  char v18; // dl
  __int64 v19; // rax
  __int64 v20; // r13
  __int64 v21; // rax
  __int64 v22; // rdi
  __int64 v23; // rax
  char v24; // dl
  __int64 v25; // rax
  char v26; // dl
  __int64 v27; // rax
  __int64 v28; // rax
  char v29; // dl
  __int64 v30; // rax
  char v31; // dl
  __int64 v32; // rax
  __int64 v33; // r10
  __int64 v34; // r9
  __int64 v35; // rdx
  __int64 v36; // rcx
  __int64 v37; // r8
  __int64 v38; // r10
  __int64 v39; // r9
  char v40; // al
  _QWORD *v41; // rax
  __int64 v42; // rdi
  __int64 v43; // r9
  __int64 v44; // rax
  char v45; // dl
  __int64 v46; // rax
  char v47; // dl
  __int64 v48; // rax
  unsigned __int64 v49; // r14
  char v50; // al
  __int64 **v51; // rcx
  __int64 v52; // r8
  __int64 v53; // [rsp+0h] [rbp-A0h]
  __int64 v54; // [rsp+8h] [rbp-98h]
  __int64 v55; // [rsp+8h] [rbp-98h]
  __int64 v56; // [rsp+8h] [rbp-98h]
  __int64 v57; // [rsp+8h] [rbp-98h]
  __int64 **v58; // [rsp+8h] [rbp-98h]
  unsigned int v59; // [rsp+14h] [rbp-8Ch]
  __int64 v60; // [rsp+18h] [rbp-88h] BYREF
  unsigned int v61; // [rsp+24h] [rbp-7Ch] BYREF
  __int64 v62; // [rsp+28h] [rbp-78h] BYREF
  _BYTE v63[112]; // [rsp+30h] [rbp-70h] BYREF

  v4 = a3 & 1;
  v60 = a1;
  v61 = 0;
  v62 = 0;
  if ( !a1 )
    return;
  v5 = *(_BYTE *)(a1 + 140);
  v59 = a3 & 0xFFFFFFFE;
  if ( v5 != 12 )
  {
    v9 = a1;
    v8 = a1;
    goto LABEL_27;
  }
  v8 = a1;
  v9 = a1;
  for ( i = 0; ; i = v62 )
  {
    if ( i )
      goto LABEL_4;
    if ( !*(_QWORD *)(v9 + 8) )
      break;
    if ( (*(_BYTE *)(v9 + 89) & 1) != 0 && *(_BYTE *)(a4 + 139)
      || v4 && (sub_8D4C10(v9, dword_4F077C4 != 2) & 1) != 0
      || *(_BYTE *)(a4 + 138)
      || (v12 = *(unsigned int (__fastcall **)(__int64, __int64 *))(a4 + 96)) != 0 && v12(v9, &v62) )
    {
      v9 = *(_QWORD *)(v60 + 160);
    }
    else
    {
      v9 = v60;
      if ( !sub_746100(v60, a4) )
        goto LABEL_26;
      v9 = *(_QWORD *)(v60 + 160);
    }
LABEL_9:
    v5 = *(_BYTE *)(v9 + 140);
    v60 = v9;
    if ( v5 != 12 )
      goto LABEL_27;
  }
  if ( sub_5D7700() || !(unsigned int)sub_746C80(v9, a4) )
  {
LABEL_4:
    v11 = v61 | *(_BYTE *)(v60 + 185) & 0x7F;
    v61 = v11;
    if ( v4 && (v11 & 1) != 0 )
    {
      v4 = 0;
      v61 = v11 & 0xFFFFFFFE;
    }
    v9 = *(_QWORD *)(v60 + 160);
    if ( *(_BYTE *)(v60 + 184) == 8 )
      v8 = *(_QWORD *)(v60 + 160);
    goto LABEL_9;
  }
  v9 = v60;
LABEL_26:
  v5 = *(_BYTE *)(v9 + 140);
LABEL_27:
  if ( v5 == 6 )
  {
    v14 = *(_QWORD *)(v9 + 160);
    if ( *(_BYTE *)(a4 + 153) && *(_BYTE *)(v14 + 140) == 12 )
    {
      do
      {
        if ( !*(_QWORD *)(v14 + 8) )
          break;
        v15 = v14;
        do
        {
          v15 = *(_QWORD *)(v15 + 160);
          v16 = *(_BYTE *)(v15 + 140);
        }
        while ( v16 == 12 );
        if ( v16 == 21 )
          break;
        v17 = v14;
        do
        {
          v17 = *(_QWORD *)(v17 + 160);
          v18 = *(_BYTE *)(v17 + 140);
        }
        while ( v18 == 12 );
        if ( !v18 )
          break;
        v19 = *(_QWORD *)(v14 + 40);
        if ( v19 )
        {
          if ( *(_BYTE *)(v19 + 28) == 3 && **(_QWORD ***)(v19 + 32) == qword_4D049B8 )
            break;
        }
        v14 = *(_QWORD *)(v14 + 160);
      }
      while ( *(_BYTE *)(v14 + 140) == 12 );
    }
    sub_74D110(v14, 1, v59, a4);
    return;
  }
  if ( v5 != 13 )
  {
    if ( v5 == 7 )
    {
      if ( a2 )
      {
        (*(void (__fastcall **)(char *, __int64))a4)(")", a4);
        v9 = v60;
      }
      sub_74BA50(v9, a4);
      if ( a1 != v8 )
        sub_74A2C0(a1, v8, a4);
      v21 = *(_QWORD *)(v60 + 168);
      if ( (*(_BYTE *)(v21 + 16) & 8) == 0 && (*(_BYTE *)(v21 + 17) & 4) == 0
        || *(_BYTE *)(a4 + 136) && *(_BYTE *)(a4 + 141) )
      {
        v22 = *(_QWORD *)(v60 + 160);
        if ( *(_BYTE *)(a4 + 153) && *(_BYTE *)(v22 + 140) == 12 )
        {
          do
          {
            if ( !*(_QWORD *)(v22 + 8) )
              break;
            v23 = v22;
            do
            {
              v23 = *(_QWORD *)(v23 + 160);
              v24 = *(_BYTE *)(v23 + 140);
            }
            while ( v24 == 12 );
            if ( v24 == 21 )
              break;
            v25 = v22;
            do
            {
              v25 = *(_QWORD *)(v25 + 160);
              v26 = *(_BYTE *)(v25 + 140);
            }
            while ( v26 == 12 );
            if ( !v26 )
              break;
            v27 = *(_QWORD *)(v22 + 40);
            if ( v27 )
            {
              if ( *(_BYTE *)(v27 + 28) == 3 && **(_QWORD ***)(v27 + 32) == qword_4D049B8 )
                break;
            }
            v22 = *(_QWORD *)(v22 + 160);
          }
          while ( *(_BYTE *)(v22 + 140) == 12 );
        }
        sub_74D110(v22, 0, v59, a4);
      }
      return;
    }
    if ( v5 != 8 )
    {
      v13 = *(__int64 **)(a1 + 104);
      if ( v13 )
      {
        while ( *((_BYTE *)v13 + 10) != 11 )
        {
          v13 = (__int64 *)*v13;
          if ( !v13 )
            return;
        }
        (*(void (__fastcall **)(char *, __int64))a4)(")", a4);
      }
      return;
    }
    if ( (unsigned int)sub_745B20(&v60, &v61, v4, a4) )
      return;
    v33 = v60;
    v34 = *(_QWORD *)(v60 + 160);
    if ( *(_BYTE *)(a4 + 153) )
    {
      while ( *(_BYTE *)(v34 + 140) == 12 )
      {
        if ( !*(_QWORD *)(v34 + 8) )
          break;
        v44 = v34;
        do
        {
          v44 = *(_QWORD *)(v44 + 160);
          v45 = *(_BYTE *)(v44 + 140);
        }
        while ( v45 == 12 );
        if ( v45 == 21 )
          break;
        v46 = v34;
        do
        {
          v46 = *(_QWORD *)(v46 + 160);
          v47 = *(_BYTE *)(v46 + 140);
        }
        while ( v47 == 12 );
        if ( !v47 )
          break;
        v48 = *(_QWORD *)(v34 + 40);
        if ( v48 )
        {
          if ( *(_BYTE *)(v48 + 28) == 3 && **(_QWORD ***)(v48 + 32) == qword_4D049B8 )
            break;
        }
        v34 = *(_QWORD *)(v34 + 160);
      }
    }
    if ( a2 )
    {
      v54 = v34;
      (*(void (__fastcall **)(char *, __int64))a4)(")", a4);
      v33 = v60;
      v34 = v54;
    }
    v53 = v34;
    v55 = v33;
    (*(void (__fastcall **)(char *, __int64))a4)("[", a4);
    sub_746940(*(_BYTE *)(v55 + 168) & 0x7F, -1, 1, a4);
    v38 = v55;
    v39 = v53;
    v40 = *(_BYTE *)(v55 + 169);
    if ( (v40 & 2) != 0 )
    {
      if ( (v40 & 0x10) == 0 || *(_BYTE *)(a4 + 143) )
      {
        (*(void (__fastcall **)(char *, __int64))a4)("*", a4);
        v39 = v53;
      }
      else if ( qword_4F04C50 )
      {
        v41 = sub_72D900(v55);
        if ( *(_BYTE *)(a4 + 136) && (v42 = v41[6]) != 0 )
        {
          sub_74C550(v42, 7u, a4);
          v39 = v53;
        }
        else
        {
          sub_747C50(v41[2], a4);
          v39 = v53;
        }
      }
      else
      {
        (*(void (__fastcall **)(const char *, __int64, __int64, __int64, __int64, __int64))a4)(
          "<expr>",
          a4,
          v35,
          v36,
          v37,
          v53);
        v39 = v53;
      }
      goto LABEL_102;
    }
    v49 = *(_QWORD *)(v55 + 176);
    if ( (v40 & 1) != 0 )
    {
      sub_747C50(*(_QWORD *)(v55 + 176), a4);
      v39 = v53;
    }
    else
    {
      if ( *(char *)(v55 + 168) < 0 )
      {
        if ( v49 )
        {
          if ( (v40 & 8) != 0 && qword_4F04C50 )
          {
            v50 = *(_BYTE *)(v49 + 176);
            v51 = (__int64 **)(v49 + 184);
            if ( v50 != 1 )
            {
              if ( (unsigned __int8)(v50 - 5) > 5u )
              {
                if ( v50 )
                {
                  MEMORY[0] = 0;
                  BUG();
                }
                v51 = (__int64 **)(v49 + 144);
              }
              else
              {
                v51 = (__int64 **)(v49 + 192);
              }
            }
            *v51 = 0;
            v58 = v51;
            *v51 = sub_72DB50(v38, 5);
            sub_748000(v49, 0, a4, (__int64)v58, v52);
            v39 = v53;
            *v58 = 0;
          }
          else
          {
            sub_748000(*(_QWORD *)(v55 + 176), 0, a4, v36, v37);
            v39 = v53;
          }
        }
        goto LABEL_102;
      }
      if ( v49 )
      {
        if ( v49 <= 9 )
        {
LABEL_122:
          v63[1] = 0;
          v63[0] = v49 + 48;
        }
        else
        {
          sub_622470(v49, v63);
          v39 = v53;
        }
        v57 = v39;
        (*(void (__fastcall **)(_BYTE *, __int64))a4)(v63, a4);
        v39 = v57;
      }
      else if ( (v40 & 0x20) != 0 )
      {
        goto LABEL_122;
      }
    }
LABEL_102:
    v56 = v39;
    (*(void (__fastcall **)(char *, __int64))a4)("]", a4);
    v43 = v56;
    if ( a1 != v8 )
    {
      sub_74A2C0(a1, v8, a4);
      v43 = v56;
    }
    sub_74D110(v43, 0, v4 | v59, a4);
    return;
  }
  v20 = *(_QWORD *)(v9 + 168);
  if ( *(_BYTE *)(a4 + 153) )
  {
    while ( *(_BYTE *)(v20 + 140) == 12 )
    {
      if ( !*(_QWORD *)(v20 + 8) )
        goto LABEL_87;
      v28 = v20;
      do
      {
        v28 = *(_QWORD *)(v28 + 160);
        v29 = *(_BYTE *)(v28 + 140);
      }
      while ( v29 == 12 );
      if ( v29 == 21 )
        goto LABEL_87;
      v30 = v20;
      do
      {
        v30 = *(_QWORD *)(v30 + 160);
        v31 = *(_BYTE *)(v30 + 140);
      }
      while ( v31 == 12 );
      if ( !v31
        || (v32 = *(_QWORD *)(v20 + 40)) != 0 && *(_BYTE *)(v32 + 28) == 3 && **(_QWORD ***)(v32 + 32) == qword_4D049B8 )
      {
LABEL_87:
        if ( !*(_BYTE *)(a4 + 136) )
          goto LABEL_53;
        goto LABEL_88;
      }
      v20 = *(_QWORD *)(v20 + 160);
    }
  }
  if ( *(_BYTE *)(a4 + 136) && *(_BYTE *)(v20 + 140) != 7 )
  {
LABEL_88:
    if ( !*(_BYTE *)(a4 + 154) )
      (*(void (__fastcall **)(char *, __int64))a4)(")", a4);
  }
LABEL_53:
  sub_74D110(v20, 1, v59, a4);
}
