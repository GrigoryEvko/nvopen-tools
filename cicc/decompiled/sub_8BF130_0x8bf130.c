// Function: sub_8BF130
// Address: 0x8bf130
//
unsigned int *__fastcall sub_8BF130(__int64 a1)
{
  unsigned int *result; // rax
  unsigned int v2; // r15d
  unsigned int v3; // ebx
  __int64 v4; // rdx
  __int64 v5; // rcx
  __int64 v6; // r8
  __int64 v7; // r9
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // r8
  __int64 v11; // r9
  __int64 v12; // rsi
  __int64 v13; // rdx
  __int64 v14; // rcx
  __int64 v15; // r8
  __int64 v16; // r9
  __int64 v17; // rdi
  __int64 v18; // rdi
  __int64 v19; // rdx
  __int64 v20; // rcx
  __int64 v21; // r8
  __int64 v22; // r9
  __int64 v23; // r14
  int v24; // eax
  int v25; // ecx
  __int64 v26; // rdi
  __int64 v27; // rax
  __int64 v28; // [rsp-10h] [rbp-240h]
  unsigned int v29; // [rsp+14h] [rbp-21Ch] BYREF
  __int64 v30; // [rsp+18h] [rbp-218h] BYREF
  _BYTE v31[528]; // [rsp+20h] [rbp-210h] BYREF

  result = &dword_4F04C44;
  v29 = 0;
  if ( dword_4F04C44 != -1 )
    return result;
  result = (unsigned int *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
  if ( (*((_BYTE *)result + 6) & 2) != 0 )
    return result;
  v2 = dword_4D04734;
  dword_4D04734 = 0;
  v3 = *(unsigned __int8 *)(*(_QWORD *)(a1 + 8) + 8LL);
  if ( (_BYTE)v3 != 17 )
  {
    if ( (unsigned __int8)(v3 - 15) > 1u )
      sub_721090();
    if ( !dword_4D04324 )
      goto LABEL_6;
LABEL_31:
    sub_684AB0(&dword_4F063F8, 0x36Bu);
    goto LABEL_6;
  }
  if ( v2 == 1 )
    v3 = 15;
  if ( dword_4D04324 )
    goto LABEL_31;
LABEL_6:
  sub_7C9660(a1);
  if ( dword_4F077C4 != 2 )
  {
    v30 = *(_QWORD *)&dword_4F063F8;
    if ( word_4F06418[0] != 1 )
      goto LABEL_9;
LABEL_8:
    if ( (unsigned __int16)sub_7BE840(0, 0) != 9 )
      goto LABEL_9;
    v12 = 0;
    v18 = 0;
    v23 = sub_7BF130(0, 0, &v29);
    v24 = v29;
    if ( v29 )
    {
LABEL_51:
      sub_7B8B50(v18, (unsigned int *)v12, v19, v20, v21, v22);
      goto LABEL_17;
    }
    if ( !v23 )
    {
      v12 = (__int64)&v30;
      v18 = 484;
      sub_6851C0(0x1E4u, &v30);
      v29 = 1;
      goto LABEL_51;
    }
    v19 = *(unsigned __int8 *)(v23 + 80);
    v25 = v19 - 4;
    if ( (unsigned __int8)(v19 - 4) <= 1u )
    {
      v27 = *(_QWORD *)(v23 + 88);
      if ( v27 && (*(_DWORD *)(v27 + 176) & 0x13000) == 0x1000 )
      {
        v12 = (unsigned __int8)v3;
        v18 = v23;
        sub_8B1370(v23, v3, &v30, 1u, 1, 0);
        goto LABEL_51;
      }
      if ( (*(_BYTE *)(v23 + 81) & 0x10) == 0 )
        goto LABEL_67;
    }
    else
    {
      if ( (*(_BYTE *)(v23 + 81) & 0x10) == 0 )
        goto LABEL_67;
      LOBYTE(v25) = (_BYTE)v19 == 17;
      LOBYTE(v12) = (_BYTE)v19 == 10;
      v12 = v25 | (unsigned int)v12;
      if ( (_BYTE)v19 == 20 || (_BYTE)v12 )
      {
        v26 = **(_QWORD **)(v23 + 64);
        if ( (unsigned __int8)(*(_BYTE *)(v26 + 80) - 4) <= 1u )
        {
          if ( *(_QWORD *)(*(_QWORD *)(v26 + 96) + 72LL) )
          {
            v20 = *(_QWORD *)(v26 + 88);
            if ( (*(_BYTE *)(v20 + 178) & 1) == 0 )
            {
              if ( (_BYTE)v19 == 10 )
              {
                v18 = v23;
LABEL_50:
                if ( !unk_4D03FD8 )
                {
                  v12 = 1;
                  if ( sub_891C80(v18, 1, 1, v3) )
                  {
                    v12 = (unsigned __int8)v3;
                    sub_8ACB90(v18, v3, &v30, 0, 1, 0, 1);
                  }
                }
                goto LABEL_51;
              }
              if ( (_BYTE)v12 )
              {
                v20 = *(_QWORD *)(v23 + 88);
                if ( v20 )
                {
                  v18 = 0;
                  do
                  {
                    v12 = *(_QWORD *)(v20 + 88);
                    if ( (*(_BYTE *)(v12 + 193) & 0x10) == 0 )
                    {
                      if ( v24 )
                        goto LABEL_83;
                      v18 = v20;
                      v24 = 1;
                    }
                    v20 = *(_QWORD *)(v20 + 8);
                  }
                  while ( v20 );
                  if ( v18 )
                    goto LABEL_50;
                }
LABEL_83:
                if ( (((_BYTE)v19 - 7) & 0xFD) != 0 )
                  goto LABEL_70;
LABEL_75:
                v18 = v23;
                if ( sub_892240(v23) )
                {
                  v22 = unk_4D03FD8;
                  if ( !unk_4D03FD8 )
                  {
                    v12 = 1;
                    v18 = v23;
                    if ( sub_891C80(v23, 1, 1, v3) )
                    {
                      v12 = v3;
                      sub_8ACB90(v23, v3, &v30, 0, 1, 0, 1);
                      v18 = v28;
                      v21 = 1;
                    }
                  }
                  goto LABEL_51;
                }
                LOBYTE(v19) = *(_BYTE *)(v23 + 80);
LABEL_68:
                if ( (_BYTE)v19 == 17 || (_BYTE)v19 == 20 )
                {
LABEL_70:
                  v12 = v23;
                  v18 = 299;
                  sub_6854E0(0x12Bu, v23);
                  v29 = 1;
                  goto LABEL_51;
                }
LABEL_79:
                v12 = v23;
                v18 = 485;
                sub_6854E0(0x1E5u, v23);
                v29 = 1;
                goto LABEL_51;
              }
            }
          }
        }
LABEL_67:
        if ( (((_BYTE)v19 - 7) & 0xFD) != 0 )
          goto LABEL_68;
        goto LABEL_75;
      }
    }
    if ( (((_BYTE)v19 - 7) & 0xFD) != 0 )
      goto LABEL_79;
    goto LABEL_75;
  }
  *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 7) |= 8u;
  v30 = *(_QWORD *)&dword_4F063F8;
  if ( word_4F06418[0] == 1 && (word_4D04A10 & 0x200) != 0 || (unsigned int)sub_7C0F00(0, 0, v4, v5, v6, v7) )
    goto LABEL_8;
LABEL_9:
  if ( (unsigned int)sub_651B00(2u) )
  {
LABEL_16:
    v12 = (unsigned __int8)v3;
    sub_8BE350((__int64)v31, (unsigned __int8)v3, 1u, &v30, v10, v11);
    goto LABEL_17;
  }
  if ( word_4F06418[0] == 1 )
  {
    if ( dword_4F077C4 != 2
      || (word_4D04A10 & 0x200) == 0 && !(unsigned int)sub_7C0F00(0, 0, v8, v9, v10, v11)
      || (unk_4D04A12 & 1) == 0 )
    {
      goto LABEL_16;
    }
  }
  else if ( word_4F06418[0] == 34
         || word_4F06418[0] == 27
         || dword_4F077C4 == 2
         && (word_4F06418[0] == 33
          || dword_4D04474 && word_4F06418[0] == 52
          || dword_4D0485C && word_4F06418[0] == 25
          || word_4F06418[0] == 156) )
  {
    goto LABEL_16;
  }
  v12 = (__int64)dword_4F07508;
  sub_6851C0(0x1E4u, dword_4F07508);
  v29 = 1;
LABEL_17:
  if ( *(_QWORD *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 456) )
    sub_87DD20(dword_4F04C40);
  if ( dword_4F077C4 == 2 )
  {
    v17 = (int)dword_4F04C40;
    *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 7) &= ~8u;
    v13 = qword_4F04C68[0];
    if ( *(_QWORD *)(qword_4F04C68[0] + 776 * v17 + 456) )
      sub_8845B0(v17);
  }
  sub_7C96B0(v29, (unsigned int *)v12, v13, v14, v15, v16);
  result = &dword_4D04734;
  dword_4D04734 = v2;
  return result;
}
