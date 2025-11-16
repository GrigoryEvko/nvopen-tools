// Function: sub_86B690
// Address: 0x86b690
//
char __fastcall sub_86B690(int a1, _QWORD *a2)
{
  _QWORD *v2; // r15
  __int64 v3; // r12
  __int64 v4; // rax
  __int64 i; // rbx
  __int64 v6; // rax
  char v7; // al
  __int64 v8; // rbx
  __int64 v9; // rbx
  unsigned __int8 v10; // di
  char v11; // dl
  char v12; // al
  unsigned __int8 v13; // dl
  __int64 v14; // rbx
  unsigned int v15; // r13d
  __int64 j; // rax
  _QWORD *v17; // rax
  char v18; // dl
  unsigned __int8 v19; // bl
  unsigned __int8 v21; // [rsp+Ch] [rbp-74h]
  unsigned __int8 v22; // [rsp+Ch] [rbp-74h]
  char v23; // [rsp+Ch] [rbp-74h]
  const __m128i *v24[14]; // [rsp+10h] [rbp-70h] BYREF

  v2 = *(_QWORD **)(qword_4F04C50 + 32LL);
  v3 = qword_4F04C68[0] + 776LL * dword_4F04C58;
  *a2 = 0;
  v4 = *(_QWORD *)(v3 + 184);
  *(_BYTE *)(v3 + 6) &= ~1u;
  *(_QWORD *)(v4 + 72) = 0;
  for ( i = v2[19]; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
    ;
  LOBYTE(v6) = *((_BYTE *)v2 + 174) - 1;
  if ( (unsigned __int8)v6 > 1u )
  {
    v7 = *((_BYTE *)v2 + 207);
    if ( v7 < 0 )
    {
      v6 = sub_71DF80((__int64)v2);
      v9 = v6;
      if ( (*(_BYTE *)(v6 + 120) & 3) == 0 )
      {
        LODWORD(v6) = sub_8DBE70(*(_QWORD *)(*(_QWORD *)(v6 + 16) + 120LL));
        if ( !(_DWORD)v6 )
        {
          v10 = 5;
          if ( dword_4D04964 )
            v10 = unk_4F07471;
          LOBYTE(v6) = sub_686750(v10, 0xB9Fu, (_DWORD *)(*v2 + 48LL), *v2, *(_QWORD *)(*(_QWORD *)(v9 + 16) + 120LL));
        }
      }
    }
    else
    {
      if ( (v7 & 0x30) == 0x10 )
        sub_695540((__int64)v2, ((*((_BYTE *)v2 + 206) >> 1) ^ 1) & 1, dword_4F07508);
      v8 = *(_QWORD *)(i + 160);
      LODWORD(v6) = sub_8D2600(v8);
      if ( !(_DWORD)v6 )
      {
        LODWORD(v6) = sub_8D3D40(v8);
        if ( !(_DWORD)v6 )
        {
          v11 = *(_BYTE *)(v8 + 140);
          if ( v11 == 12 )
          {
            v6 = v8;
            do
            {
              v6 = *(_QWORD *)(v6 + 160);
              v11 = *(_BYTE *)(v6 + 140);
            }
            while ( v11 == 12 );
          }
          if ( v11 )
          {
            if ( (_QWORD *)unk_4F07290 == v2 && (unsigned int)sub_8D2780(v8) )
            {
              for ( j = v8; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
                ;
              if ( *(_BYTE *)(j + 160) == 5 )
              {
                v24[0] = (const __m128i *)sub_724DC0();
                sub_72BB40(v8, v24[0]);
                v17 = sub_73A720(v24[0], (__int64)v24[0]->m128i_i64);
                v18 = 5;
                *a2 = v17;
                if ( a1 )
                {
                  if ( dword_4F077C4 == 2 || (v18 = 4, unk_4F07778 > 199900) )
                  {
                    LOBYTE(v6) = (unsigned __int8)sub_724E30((__int64)v24);
                    return v6;
                  }
                }
                else if ( dword_4D04964 && (dword_4F077C4 == 2 || unk_4F07778 > 199900) )
                {
                  v18 = unk_4F07471;
                }
                v23 = v18;
                sub_724E30((__int64)v24);
                v13 = v23;
LABEL_30:
                LOBYTE(v6) = v13 <= 5u;
LABEL_31:
                if ( (a1 & 1) != 0 && (_BYTE)v6 )
                  goto LABEL_33;
LABEL_35:
                v14 = *v2;
                if ( (*(_BYTE *)(*v2 + 81LL) & 0x20) == 0
                  || (v22 = v13, sub_878710(*v2, v24), v13 = v22, (*(_BYTE *)(v14 + 81) & 0x20) == 0)
                  || (LODWORD(v6) = sub_878690(v24), v13 = v22, !(_DWORD)v6) )
                {
                  v21 = v13;
                  v15 = a1 == 0 ? 117 : 940;
                  LOBYTE(v6) = sub_685440(v13, v15, v14);
                  if ( (*((_BYTE *)v2 + 193) & 2) != 0 )
                  {
                    v6 = *(_QWORD *)(qword_4F04C50 + 32LL);
                    if ( *(_BYTE *)(v6 + 174) != 1 )
                    {
                      LODWORD(v6) = sub_67D370((int *)v15, v21, dword_4F07508);
                      if ( (_DWORD)v6 )
                        *(_BYTE *)(v3 + 13) |= 0x10u;
                    }
                  }
                }
                return v6;
              }
            }
            if ( (*((_BYTE *)v2 + 193) & 2) == 0 || dword_4D0488C )
              goto LABEL_44;
            if ( word_4D04898 )
            {
              if ( !(_DWORD)qword_4F077B4 )
              {
                if ( !dword_4F077BC )
                  goto LABEL_25;
LABEL_67:
                if ( (*((_BYTE *)v2 + 195) & 8) == 0 )
                  goto LABEL_25;
                v12 = *(_BYTE *)(v3 + 13);
                if ( (v12 & 0x10) == 0 )
                {
                  v13 = 5;
                  goto LABEL_27;
                }
                goto LABEL_51;
              }
              if ( qword_4F077A0 <= 0x765Bu )
                goto LABEL_25;
              if ( sub_729F80(dword_4F063F8) )
              {
LABEL_44:
                if ( !dword_4D04964 || (a1 & 1) != 0 )
                {
                  if ( dword_4F077C4 == 2 )
                    goto LABEL_51;
                }
                else if ( dword_4F077C4 == 2 )
                {
                  v13 = unk_4F07471;
                  goto LABEL_35;
                }
                if ( unk_4F07778 > 199900 && (a1 & 1) == 0 )
                {
                  v13 = 5;
                  if ( dword_4F077C0 )
                    goto LABEL_35;
                  goto LABEL_64;
                }
                if ( (*(_BYTE *)(unk_4D03B98 + 4LL) & 0x40) == 0 )
                {
                  LOBYTE(v6) = 1;
                  v13 = 4;
                  goto LABEL_31;
                }
LABEL_51:
                LOBYTE(v6) = 1;
                v13 = 5;
                goto LABEL_31;
              }
            }
            if ( !dword_4F077BC || (_DWORD)qword_4F077B4 )
            {
LABEL_25:
              v12 = *(_BYTE *)(v3 + 13);
              if ( (v12 & 0x10) == 0 )
              {
                v13 = 7;
LABEL_27:
                if ( !a1 )
                  goto LABEL_35;
                if ( *((_BYTE *)v2 + 174) != 1 && (v12 & 8) == 0 )
                {
                  v19 = v13;
                  sub_684AA0(v13, 0x953u, &dword_4F063F8);
                  LODWORD(v6) = sub_67D370((int *)0x953, v19, dword_4F07508);
                  if ( (_DWORD)v6 )
                    *(_BYTE *)(v3 + 13) |= 0x10u;
                  v13 = 3;
LABEL_33:
                  if ( !HIDWORD(qword_4F5FD78) || (*((_BYTE *)v2 + 195) & 8) != 0 )
                    return v6;
                  goto LABEL_35;
                }
                goto LABEL_30;
              }
LABEL_64:
              v13 = 7;
              goto LABEL_35;
            }
            goto LABEL_67;
          }
        }
      }
    }
  }
  return v6;
}
