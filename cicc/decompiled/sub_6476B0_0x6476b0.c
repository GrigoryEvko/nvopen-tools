// Function: sub_6476B0
// Address: 0x6476b0
//
__int64 __fastcall sub_6476B0(__int64 a1, _QWORD *a2, __int64 a3, unsigned __int8 a4)
{
  __int64 *v4; // r14
  __int64 v5; // r12
  __int64 v7; // rbx
  __int64 v8; // rax
  int v9; // ecx
  int v10; // r13d
  __int64 j; // rax
  __int64 v12; // rax
  int v13; // eax
  unsigned __int8 v14; // r10
  unsigned int v15; // r9d
  int v16; // esi
  char i; // al
  char v18; // al
  __int64 v20; // rax
  int v21; // eax
  __int64 v22; // rax
  __int64 v23; // rax
  _BOOL4 v24; // edx
  unsigned __int8 v25; // di
  __int64 v26; // rsi
  int v27; // eax
  int v28; // eax
  char v29; // r10
  int v30; // eax
  int v31; // ecx
  char v32; // r10
  int v33; // eax
  char v34; // r10
  unsigned int v35; // eax
  char v36; // r10
  int v37; // eax
  int v38; // eax
  int v39; // eax
  int v40; // [rsp+8h] [rbp-48h]
  int v41; // [rsp+8h] [rbp-48h]
  int v42; // [rsp+Ch] [rbp-44h]
  char v43; // [rsp+Ch] [rbp-44h]
  char v44; // [rsp+Ch] [rbp-44h]
  char v46; // [rsp+18h] [rbp-38h]
  int v47; // [rsp+18h] [rbp-38h]
  int v48; // [rsp+18h] [rbp-38h]
  char v49; // [rsp+18h] [rbp-38h]
  char v50; // [rsp+18h] [rbp-38h]
  unsigned int v52; // [rsp+1Ch] [rbp-34h]

  v4 = *(__int64 **)(a1 + 88);
  v5 = *v4;
  if ( *v4 == a3 )
    return 1;
  v7 = a3;
  if ( v5 )
  {
    if ( a3 )
    {
      if ( dword_4F07588 )
      {
        v8 = *(_QWORD *)(v5 + 32);
        if ( *(_QWORD *)(a3 + 32) == v8 )
        {
          if ( v8 )
            return 1;
        }
      }
    }
  }
  v46 = *(_BYTE *)(a1 + 80);
  if ( v46 != 15 )
  {
    if ( !(unsigned int)sub_8DED30(v5, a3, 5) )
      goto LABEL_10;
    goto LABEL_22;
  }
  if ( dword_4F077C4 == 2 )
  {
    if ( (unsigned int)sub_8DED30(v5, a3, 0x40000) )
      goto LABEL_22;
    if ( !(unsigned int)sub_8DED30(v5, v7, 1314824) )
      goto LABEL_10;
    v9 = 0;
    if ( dword_4F077BC )
    {
      if ( !(_DWORD)qword_4F077B4 )
      {
        if ( !qword_4F077A8 )
          goto LABEL_48;
        goto LABEL_47;
      }
    }
    else if ( !(_DWORD)qword_4F077B4 )
    {
      goto LABEL_48;
    }
    if ( dword_4F077C4 != 2 || !qword_4F077A0 )
      goto LABEL_48;
LABEL_47:
    if ( *(_BYTE *)(v5 + 140) == 7 )
    {
      v38 = sub_729F80(*(unsigned int *)(a1 + 48));
      v9 = 0;
      if ( v38 )
      {
        v39 = sub_8DED30(v5, v7, 1314816);
        v9 = 0;
        if ( v39 )
        {
          *(_QWORD *)(a1 + 48) = *a2;
LABEL_22:
          *v4 = sub_8D79B0(v5, v7);
          for ( i = *(_BYTE *)(v5 + 140); i == 12; i = *(_BYTE *)(v5 + 140) )
            v5 = *(_QWORD *)(v5 + 160);
          v15 = 0;
          if ( !i )
            return v15;
          while ( 1 )
          {
            v18 = *(_BYTE *)(v7 + 140);
            if ( v18 != 12 )
              break;
            v7 = *(_QWORD *)(v7 + 160);
          }
          v15 = 0;
          if ( !v18 )
            return v15;
          return 1;
        }
      }
    }
LABEL_48:
    v10 = 1;
    goto LABEL_11;
  }
  if ( (unsigned int)sub_8DED30(v5, a3, 1) )
    goto LABEL_22;
LABEL_10:
  v9 = 1;
  v10 = 0;
LABEL_11:
  if ( unk_4D0436C )
  {
    if ( dword_4F04C5C )
    {
      v42 = v9;
      v21 = sub_8D3410(v7);
      v9 = v42;
      if ( !v21 )
      {
        v22 = **(_QWORD **)(*(_QWORD *)(a1 + 88) + 8LL);
        if ( v22 )
        {
          v23 = *(_QWORD *)(*(_QWORD *)v22 + 24LL);
          if ( v23 )
          {
            v24 = 0;
            do
            {
              v25 = *(_BYTE *)(v23 + 80);
              if ( dword_4F04BA0[v25] == 2 )
              {
                if ( *(_DWORD *)(v23 + 40) == *(_DWORD *)qword_4F04C68[0] )
                {
                  if ( (unsigned __int8)(v25 - 4) > 2u )
                  {
                    if ( v25 != 3 )
                    {
                      if ( v46 == 15 )
                      {
                        if ( v25 != 11 )
                          goto LABEL_100;
                      }
                      else if ( v25 != 7 )
                      {
                        goto LABEL_69;
                      }
                      if ( !v24 )
                        goto LABEL_14;
                      v34 = 5;
                      v41 = v42;
                      if ( a4 <= 5u )
                        v34 = a4;
                      v44 = v34;
                      v35 = sub_6416D0(v5, v7);
                      v14 = v44;
                      v31 = v41;
                      v15 = v35;
                      if ( !v35 )
                        goto LABEL_19;
LABEL_78:
                      v15 = v31;
                      if ( v46 != 15 )
                        goto LABEL_72;
                      goto LABEL_19;
                    }
                    if ( !*(_BYTE *)(v23 + 104) )
                    {
                      if ( v46 != 15 )
                      {
LABEL_69:
                        v29 = 5;
                        if ( a4 <= 5u )
                          v29 = a4;
                        v49 = v29;
                        v30 = sub_6416D0(v5, v7);
                        v14 = v49;
                        v31 = v42;
                        v15 = 0;
                        if ( !v30 )
                          goto LABEL_89;
LABEL_72:
                        *v4 = v7;
                        v15 = v31;
                        goto LABEL_19;
                      }
LABEL_100:
                      v36 = 5;
                      if ( a4 <= 5u )
                        v36 = a4;
                      v50 = v36;
                      v37 = sub_6416D0(v5, v7);
                      v14 = v50;
                      v15 = 0;
                      if ( v37 )
                      {
                        v15 = v42;
                        goto LABEL_19;
                      }
                      goto LABEL_98;
                    }
                  }
                }
                else if ( !v24 )
                {
                  if ( v46 == 15 )
                  {
                    v24 = v25 != 11;
                  }
                  else if ( v25 == 7 )
                  {
                    if ( *(_QWORD *)(v23 + 88) )
                      v24 = (*(_BYTE *)(v23 + 81) & 2) != 0;
                  }
                  else
                  {
                    v24 = 1;
                  }
                }
              }
              v23 = *(_QWORD *)(v23 + 8);
            }
            while ( v23 );
          }
        }
        v32 = 5;
        v40 = v42;
        if ( a4 <= 5u )
          v32 = a4;
        v43 = v32;
        v33 = sub_6416D0(v5, v7);
        v14 = v43;
        v31 = v40;
        v15 = 0;
        if ( !v33 )
        {
          if ( v46 == 15 )
          {
LABEL_98:
            *(_BYTE *)(v4[1] + 202) |= 0x40u;
            v4[1] = 0;
          }
          else
          {
LABEL_89:
            *(_BYTE *)(v4[1] + 173) |= 1u;
            v4[1] = 0;
          }
          *v4 = v7;
          goto LABEL_19;
        }
        goto LABEL_78;
      }
      v27 = sub_6416D0(v5, v7);
      v9 = v42;
      if ( v27 )
      {
        v14 = 5;
        v15 = v42;
        if ( a4 <= 5u )
          v14 = a4;
        goto LABEL_19;
      }
    }
    else if ( v46 == 15 )
    {
      v48 = v9;
      v28 = sub_6416D0(v5, v7);
      v9 = v48;
      if ( v28 )
      {
        v14 = 5;
        *v4 = v7;
        v15 = v48;
        if ( a4 <= 5u )
          v14 = a4;
        goto LABEL_19;
      }
    }
  }
LABEL_14:
  v47 = v9;
  if ( (unsigned int)sub_8D2310(v7) )
  {
    for ( j = v7; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
      ;
    v12 = sub_8D4940(*(_QWORD *)(j + 160));
    v13 = sub_8D3EA0(v12);
    if ( v13 )
    {
      *v4 = v7;
      v14 = a4;
      v15 = 0;
    }
    else
    {
      v15 = dword_4F04C5C;
      if ( dword_4F04C5C )
      {
        v26 = v4[1];
        if ( !v26 )
          goto LABEL_80;
        if ( (*(_BYTE *)(v26 + 193) & 2) != 0 )
        {
          v14 = a4;
          v15 = 0;
          goto LABEL_19;
        }
        if ( (*(_BYTE *)(v26 + 193) & 0x20) != 0 )
        {
LABEL_80:
          v14 = a4;
          v15 = v47;
        }
        else
        {
          v14 = a4;
          if ( !*(_DWORD *)(v26 + 160) )
            v13 = v47;
          v15 = v13;
        }
      }
      else
      {
        *v4 = v7;
        v14 = a4;
      }
    }
  }
  else
  {
    v20 = sub_72C930(v7);
    v14 = a4;
    v15 = 0;
    *v4 = v20;
  }
LABEL_19:
  if ( a4 != 3 )
  {
    v16 = -(v10 == 0);
    v52 = v15;
    LOBYTE(v16) = v16 & 0x4E;
    sub_6853B0(v14, (unsigned int)(v16 + 337), a2, a1);
    return v52;
  }
  return v15;
}
