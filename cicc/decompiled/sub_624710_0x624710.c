// Function: sub_624710
// Address: 0x624710
//
__int64 __fastcall sub_624710(__int64 a1, __int64 *a2, __int64 *a3, __int64 a4, int a5)
{
  __int64 v5; // r13
  __int64 v7; // r8
  char v8; // r15
  __int64 result; // rax
  __int64 i; // rbx
  __int64 v11; // rdi
  int v12; // edx
  __int64 v13; // rax
  _BOOL4 v14; // edx
  __int64 v15; // rbx
  __int64 v16; // rdi
  char v17; // al
  __int64 v18; // rax
  char v19; // dl
  __int64 v20; // rax
  __int64 v21; // r13
  __int64 v22; // rax
  __int64 v23; // rdx
  __int64 v24; // rax
  __int64 v25; // rbx
  char v26; // al
  __int64 v27; // r8
  int v28; // eax
  __int64 v29; // r8
  int v30; // eax
  __int64 v31; // r8
  int v32; // edx
  char v33; // al
  __int64 v34; // rdi
  _DWORD *v35; // rdx
  __int64 v36; // r8
  int v37; // eax
  __int64 j; // rax
  char v39; // al
  int v40; // eax
  __int64 v41; // rax
  int v42; // edx
  int v43; // [rsp+Ch] [rbp-64h]
  int v44; // [rsp+Ch] [rbp-64h]
  int v46; // [rsp+10h] [rbp-60h]
  __int64 v47; // [rsp+10h] [rbp-60h]
  __int64 v48; // [rsp+10h] [rbp-60h]
  __int64 v49; // [rsp+10h] [rbp-60h]
  __int64 v50; // [rsp+10h] [rbp-60h]
  int v51; // [rsp+10h] [rbp-60h]
  __int64 v52; // [rsp+10h] [rbp-60h]
  int v53; // [rsp+10h] [rbp-60h]
  int v55; // [rsp+24h] [rbp-4Ch] BYREF
  __int64 v56; // [rsp+28h] [rbp-48h] BYREF
  __int64 v57; // [rsp+30h] [rbp-40h] BYREF
  __int64 v58[7]; // [rsp+38h] [rbp-38h] BYREF

  v5 = a1;
  v7 = *a3;
  v58[0] = 0;
  if ( !v7 )
  {
    *a2 = a1;
    v7 = a1;
    *a3 = a1;
    goto LABEL_3;
  }
  v8 = *(_BYTE *)(v7 + 140);
  if ( v8 )
  {
    v43 = qword_4D0495C | HIDWORD(qword_4D0495C);
    if ( qword_4D0495C )
      v43 = sub_623FC0(a1, &v57, &v56, v58) != 0;
    if ( v8 == 8 )
    {
      for ( i = a1; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      if ( (unsigned int)sub_8D25A0(i) )
      {
        if ( *(_QWORD *)(i + 128) )
          goto LABEL_105;
        v11 = i;
        do
          v11 = sub_8D48B0(v11, &v55);
        while ( v11 );
        if ( !v55 )
        {
LABEL_105:
          if ( unk_4D042A4 && (unsigned int)sub_8D3A70(i) && (*(_BYTE *)(i + 179) & 8) != 0 )
          {
            if ( !HIDWORD(qword_4F077B4) )
            {
              sub_6851C0(1029, dword_4F07508);
              v44 = 1;
LABEL_16:
              if ( a5 && dword_4F077BC || !(unsigned int)sub_8D5830(i) )
              {
                if ( (unsigned int)sub_8D2BE0(i) )
                {
                  sub_685360(3414, dword_4F07508);
                  v12 = 0;
                  goto LABEL_22;
                }
              }
              else
              {
                sub_5EB950(8u, 604, i, (__int64)dword_4F07508);
              }
              if ( v44 )
              {
LABEL_21:
                v12 = 0;
LABEL_22:
                v46 = v12;
                v13 = sub_72C930();
                v14 = v46;
                v7 = v13;
                goto LABEL_23;
              }
LABEL_75:
              *(_QWORD *)(*a3 + 160) = v5;
              v15 = *a3;
              *a3 = v5;
              if ( (*(_BYTE *)(a4 + 128) & 2) != 0 )
              {
                v7 = v5;
                goto LABEL_3;
              }
LABEL_26:
              if ( !(unsigned int)sub_8D2310(v5) && !*(_QWORD *)(v5 + 128) )
              {
                v16 = v5;
                do
                  v16 = sub_8D48B0(v16, &v55);
                while ( v16 );
                if ( v55 )
                {
                  while ( 1 )
                  {
                    v17 = *(_BYTE *)(v5 + 140);
                    if ( v17 != 12 )
                      break;
                    v5 = *(_QWORD *)(v5 + 160);
                  }
                  if ( v17 )
                  {
LABEL_35:
                    v7 = *a3;
                    goto LABEL_3;
                  }
                }
              }
LABEL_41:
              while ( ((v8 - 6) & 0xFD) == 0 || (unsigned __int8)(v8 - 12) <= 1u )
              {
                sub_8D6090(v15);
                v19 = *(_BYTE *)(v15 + 140);
                if ( v19 == 12 )
                {
                  v20 = v15;
                  do
                  {
                    v20 = *(_QWORD *)(v20 + 160);
                    v19 = *(_BYTE *)(v20 + 140);
                  }
                  while ( v19 == 12 );
                }
                if ( !v19 )
                  *a3 = v15;
                v21 = *a2;
                if ( *a2 == v15 )
                  break;
                if ( v21 )
                {
                  if ( dword_4F07588 )
                  {
                    v22 = *(_QWORD *)(v15 + 32);
                    if ( *(_QWORD *)(v21 + 32) == v22 )
                    {
                      if ( v22 )
                        break;
                    }
                  }
                }
                while ( 1 )
                {
                  v24 = sub_8D48B0(v21, 0);
                  if ( v15 == v24 )
                    break;
                  if ( v24 )
                  {
                    if ( dword_4F07588 )
                    {
                      v23 = *(_QWORD *)(v24 + 32);
                      if ( *(_QWORD *)(v15 + 32) == v23 )
                      {
                        if ( v23 )
                          break;
                      }
                    }
                  }
                  v21 = v24;
                }
                v8 = *(_BYTE *)(v21 + 140);
                v15 = v21;
              }
              goto LABEL_35;
            }
            sub_684B30(1717, dword_4F07508);
          }
          v44 = 0;
          goto LABEL_16;
        }
      }
      if ( (unsigned int)sub_8D2E30(i) )
        goto LABEL_75;
      if ( *(_BYTE *)(i + 140) == 8
        && ((*(_WORD *)(i + 168) & 0x180) != 0
         || *(_QWORD *)(i + 176)
         || dword_4F077C0
         || (*(_BYTE *)(i + 169) & 0x20) != 0) )
      {
        v25 = sub_8D40F0(i);
        if ( !v25 )
          goto LABEL_75;
        while ( *(_BYTE *)(v25 + 140) == 12 )
          v25 = *(_QWORD *)(v25 + 160);
        if ( !(unsigned int)sub_8D23B0(v25) )
          goto LABEL_75;
        v26 = *(_BYTE *)(v25 + 140);
        if ( (unsigned __int8)(v26 - 9) > 2u )
        {
          if ( v26 != 2 )
            goto LABEL_75;
          v7 = v5;
          v14 = (*(_BYTE *)(v25 + 161) & 8) != 0;
LABEL_23:
          *(_QWORD *)(*a3 + 160) = v7;
          v15 = *a3;
          *a3 = v7;
          if ( (*(_BYTE *)(a4 + 128) & 2) != 0 )
            goto LABEL_3;
          if ( v14 )
            goto LABEL_41;
          goto LABEL_25;
        }
LABEL_117:
        v7 = v5;
        v14 = 1;
        goto LABEL_23;
      }
      if ( (unsigned int)sub_8D3D10(i) && !sub_8D4870(i) )
        goto LABEL_75;
      v32 = sub_8D3D40(i);
      if ( v32 )
        goto LABEL_75;
      v33 = *(_BYTE *)(i + 140);
      if ( (unsigned __int8)(v33 - 9) <= 2u )
      {
        if ( dword_4F077C4 == 2 )
        {
          v34 = i;
        }
        else
        {
          v34 = i;
          if ( dword_4D04964 )
          {
            if ( !(unsigned int)sub_8D23B0(i) )
              goto LABEL_75;
            *(_BYTE *)(i + 180) |= 0x20u;
LABEL_116:
            sub_684AC0(byte_4F07472[0], 731);
            v12 = 1;
            if ( byte_4F07472[0] == 8 )
              goto LABEL_22;
            goto LABEL_117;
          }
        }
        if ( !(unsigned int)sub_8D23B0(v34) )
          goto LABEL_75;
        *(_BYTE *)(i + 180) |= 0x20u;
        if ( !a5 || !dword_4F077BC )
          sub_880320(i, 3, *a3, 6, dword_4F07508);
        goto LABEL_117;
      }
      if ( v33 != 2 || (*(_BYTE *)(i + 161) & 8) == 0 )
      {
        v51 = v32;
        if ( v43 )
        {
          sub_6854E0(473, v58[0]);
          v12 = v51;
          goto LABEL_22;
        }
        if ( (unsigned int)sub_8D2310(i) )
        {
          sub_6851C0(88, dword_4F07508);
          v12 = 0;
          goto LABEL_22;
        }
        if ( (unsigned int)sub_8D2600(i) )
        {
          sub_6851C0(89, dword_4F07508);
          v12 = 0;
          goto LABEL_22;
        }
        v42 = sub_8D32E0(i);
        if ( v42 )
        {
          sub_6851C0(251, dword_4F07508);
          v12 = 0;
          goto LABEL_22;
        }
        if ( !*(_BYTE *)(i + 140) )
          goto LABEL_21;
        if ( (*(_BYTE *)(a4 + 128) & 2) == 0 )
        {
          v53 = v42;
          sub_6851C0(98, dword_4F07508);
          v12 = v53;
          goto LABEL_22;
        }
        goto LABEL_75;
      }
      if ( !(unsigned int)sub_8D23B0(i) )
        goto LABEL_75;
      if ( dword_4F077C4 == 2 || !dword_4D04964 )
        goto LABEL_117;
      goto LABEL_116;
    }
    if ( (unsigned int)sub_8D2EF0(*a3) )
    {
      if ( v43 )
      {
        v18 = sub_73F0A0(v57, v56, 0);
        sub_73C230(v18, *a3);
        v7 = v57;
        v15 = *a3;
        *a3 = v57;
        if ( (*(_BYTE *)(a4 + 128) & 2) != 0 )
          goto LABEL_3;
        v8 = 13;
        goto LABEL_41;
      }
      if ( (unsigned int)sub_8D32E0(a1) )
      {
        sub_6851C0(248, dword_4F07508);
        v5 = sub_72C930();
      }
      if ( (unsigned int)sub_8D2310(v5) )
        sub_6236C0(a2, a3);
      v7 = v5;
      *(_QWORD *)(*a3 + 160) = v5;
      goto LABEL_82;
    }
    if ( (unsigned int)sub_8D32E0(*a3) )
    {
      for ( ; *(_BYTE *)(a1 + 140) == 12; a1 = *(_QWORD *)(a1 + 160) )
        ;
      if ( (unsigned int)sub_8D2600(a1) )
      {
        sub_6851C0(250, dword_4F07508);
      }
      else
      {
        v27 = v5;
        if ( !v43 )
        {
LABEL_92:
          v47 = v27;
          v28 = sub_8D2310(v27);
          v7 = v47;
          if ( v28 )
          {
            sub_6236C0(a2, a3);
            v7 = v47;
          }
          *(_QWORD *)(*a3 + 160) = v7;
          goto LABEL_82;
        }
        sub_6854E0(473, v58[0]);
      }
      v27 = sub_72C930();
      goto LABEL_92;
    }
    if ( (unsigned int)sub_8D3D10(*a3) )
    {
      if ( v43 )
      {
        sub_6854E0(473, v58[0]);
      }
      else if ( (unsigned int)sub_8D2600(a1) )
      {
        sub_6851C0(2335, dword_4F07508);
      }
      else
      {
        v37 = sub_8D32E0(a1);
        v29 = a1;
        if ( !v37 )
          goto LABEL_101;
        sub_6851C0(2336, dword_4F07508);
      }
      v29 = sub_72C930();
LABEL_101:
      v48 = v29;
      v30 = sub_8D2310(v29);
      v31 = v48;
      if ( v30 )
      {
        sub_6236C0(a2, a3);
        v31 = v48;
      }
      v49 = v31;
      sub_73F060(*a3, v31);
      v7 = v49;
LABEL_82:
      v15 = *a3;
      *a3 = v7;
      if ( v8 == 7 || (*(_BYTE *)(a4 + 128) & 2) != 0 )
        goto LABEL_3;
      if ( v8 == 6 || v8 == 13 )
        goto LABEL_41;
LABEL_25:
      v5 = v7;
      goto LABEL_26;
    }
    if ( v43 )
    {
      sub_6854E0(473, v58[0]);
    }
    else
    {
      if ( (*(_BYTE *)(a4 + 123) & 0x10) != 0 )
        v35 = (_DWORD *)(a4 + 56);
      else
        v35 = dword_4F07508;
      if ( sub_624600(a1, a4, (__int64)v35) )
      {
        if ( dword_4F077C4 != 1 || !(unsigned int)sub_8D2A90(a1) )
          goto LABEL_126;
        for ( j = a1; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
          ;
        if ( *(_BYTE *)(j + 160) == 2 )
          v36 = sub_72C610(4);
        else
LABEL_126:
          v36 = a1;
        if ( dword_4F077C0 )
        {
          if ( (*(_BYTE *)(v36 + 140) & 0xFB) == 8 )
          {
            v52 = v36;
            v39 = sub_8D4C10(v36, dword_4F077C4 != 2);
            v36 = v52;
            if ( (v39 & 2) != 0 )
            {
              v40 = sub_8D2600(v52);
              v36 = v52;
              if ( !v40 )
              {
                v41 = sub_8D21F0(v52);
                v36 = sub_73C570(v41, 0, -1);
              }
            }
          }
        }
        goto LABEL_129;
      }
    }
    v36 = sub_72C930();
LABEL_129:
    v50 = v36;
    *(_QWORD *)(*a3 + 160) = v36;
    sub_7325D0(*a3, dword_4F07508);
    v7 = v50;
    goto LABEL_82;
  }
LABEL_3:
  result = sub_8D4940(v7);
  *a3 = result;
  return result;
}
