// Function: sub_5D7860
// Address: 0x5d7860
//
__int64 __fastcall sub_5D7860(__int64 *a1, __int64 a2)
{
  FILE *v4; // rsi
  __int64 v5; // rbx
  char v6; // al
  __int64 *v7; // r12
  __int64 v8; // rax
  __int64 v9; // r15
  FILE *v10; // rsi
  __int64 *v11; // r10
  __int64 v12; // rcx
  __int64 v13; // rax
  __int64 v14; // rax
  int v16; // edi
  char *v17; // r15
  __int64 v18; // r15
  int v19; // edi
  char *v20; // rbx
  __int64 v21; // rax
  int v22; // eax
  int v23; // edi
  char *v24; // r15
  int v25; // edi
  char *v26; // rbx
  int v27; // edi
  char *v28; // r15
  int v29; // eax
  char v30; // al
  char *v31; // r15
  __int64 *v32; // [rsp+10h] [rbp-60h]
  __int64 v33; // [rsp+10h] [rbp-60h]
  __int64 *v34; // [rsp+10h] [rbp-60h]
  __int64 *v35; // [rsp+10h] [rbp-60h]
  __int64 *v36; // [rsp+10h] [rbp-60h]
  char v37; // [rsp+1Fh] [rbp-51h]
  _BYTE v38[8]; // [rsp+20h] [rbp-50h] BYREF
  __int64 *v39; // [rsp+28h] [rbp-48h]
  int v40; // [rsp+30h] [rbp-40h]

  v40 = 0;
  v37 = byte_4CF7D6F;
  v39 = (__int64 *)*a1;
  sub_750500(v38, &qword_4CF7CE0);
  if ( a2 )
  {
    v4 = stream;
    v5 = *(_QWORD *)(a2 + 40);
    putc(40, stream);
    v6 = *((_BYTE *)a1 + 16);
    ++dword_4CF7F40;
    if ( (v6 & 2) == 0 || (v6 & 4) != 0 )
    {
      v10 = stream;
      if ( v5 )
      {
        while ( 1 )
        {
          sub_5D6390(v5);
          v5 = *(_QWORD *)(v5 + 112);
          v10 = stream;
          if ( !v5 )
            break;
          putc(44, stream);
          ++dword_4CF7F40;
          putc(32, stream);
          ++dword_4CF7F40;
        }
      }
      goto LABEL_30;
    }
  }
  else
  {
    v4 = stream;
    byte_4CF7D6F = 1;
    putc(40, stream);
    v6 = *((_BYTE *)a1 + 16);
    ++dword_4CF7F40;
    if ( (v6 & 2) == 0 )
      goto LABEL_39;
    v5 = 0;
  }
  v7 = (__int64 *)*a1;
  if ( !*a1 )
  {
    v10 = stream;
    if ( (v6 & 1) == 0 )
    {
      v25 = 118;
      v26 = "oid";
      do
      {
        ++v26;
        putc(v25, stream);
        v25 = *(v26 - 1);
      }
      while ( *(v26 - 1) );
      dword_4CF7F40 += 4;
      v10 = stream;
    }
    goto LABEL_30;
  }
  while ( 1 )
  {
    if ( a2 )
    {
      sub_5D45D0((unsigned int *)(v5 + 64));
      if ( *(_BYTE *)(v5 + 136) == 5 )
        sub_5D57F0(v5);
      if ( qword_4CF7EB8 == stream )
      {
        v8 = a1[1];
        if ( !v8 )
          goto LABEL_11;
        if ( (*(_BYTE *)(v8 + 198) & 0x30) != 0x10 )
        {
LABEL_10:
          if ( (*(_BYTE *)(v8 + 197) & 0x60) == 0x20 && (unsigned int)sub_8D3A70(v7[1]) )
          {
            v18 = *(_QWORD *)(v5 + 120);
            sub_74A390(v18, 0, 0, 0, 0, &qword_4CF7CE0);
            sub_74D110(v18, 0, 0, &qword_4CF7CE0);
            putc(38, stream);
            ++dword_4CF7F40;
            sub_5D6390(v5);
            goto LABEL_12;
          }
LABEL_11:
          v9 = *(_QWORD *)(v5 + 120);
          sub_74A390(v9, 0, 1, 0, 0, &qword_4CF7CE0);
          sub_5D6390(v5);
          sub_74D110(v9, 0, 0, &qword_4CF7CE0);
LABEL_12:
          sub_74F860(v5, 1, &qword_4CF7CE0);
          v5 = *(_QWORD *)(v5 + 112);
          goto LABEL_13;
        }
        if ( (unsigned int)sub_8D2FF0(*(_QWORD *)(v5 + 120), v4) )
        {
          v27 = 95;
          v28 = "_text__ ";
          do
          {
            ++v28;
            putc(v27, stream);
            v27 = *(v28 - 1);
          }
          while ( *(v28 - 1) );
        }
        else
        {
          if ( !(unsigned int)sub_8D3030(*(_QWORD *)(v5 + 120)) )
            goto LABEL_9;
          v16 = 95;
          v17 = "_surf__ ";
          do
          {
            ++v17;
            putc(v16, stream);
            v16 = *(v17 - 1);
          }
          while ( *(v17 - 1) );
        }
        dword_4CF7F40 += 9;
      }
LABEL_9:
      v8 = a1[1];
      if ( !v8 )
        goto LABEL_11;
      goto LABEL_10;
    }
    v11 = v7;
    if ( (v7[4] & 0x100200) == 0x100000 || (v11 = 0, !unk_4F068C4) || unk_4F077C4 == 2 )
    {
      if ( qword_4CF7EB8 != stream )
        goto LABEL_19;
    }
    else
    {
      v11 = 0;
      if ( unk_4F07778 >= 199901 )
        v11 = v7;
      if ( qword_4CF7EB8 != stream )
        goto LABEL_19;
    }
    v21 = a1[1];
    if ( v21 && (*(_BYTE *)(v21 + 198) & 0x30) == 0x10 )
    {
      v34 = v11;
      v22 = sub_8D2FF0(v7[1], v4);
      v11 = v34;
      if ( v22 )
      {
        v23 = 95;
        v24 = "_text__ ";
        do
        {
          ++v24;
          v35 = v11;
          putc(v23, stream);
          v23 = *(v24 - 1);
          v11 = v35;
        }
        while ( *(v24 - 1) );
      }
      else
      {
        v29 = sub_8D3030(v7[1]);
        v11 = v34;
        if ( !v29 )
          goto LABEL_19;
        v30 = 95;
        v31 = "_surf__ ";
        do
        {
          ++v31;
          v36 = v11;
          putc(v30, stream);
          v30 = *(v31 - 1);
          v11 = v36;
        }
        while ( v30 );
      }
      dword_4CF7F40 += 9;
    }
LABEL_19:
    v32 = v11;
    sub_5D7700();
    v12 = (*((_DWORD *)v7 + 8) >> 11) & 0x7F;
    if ( v32 )
    {
      v33 = v7[1];
      sub_74A390(v33, 0, 1, v12, 0, &qword_4CF7CE0);
      sub_5D34A0();
    }
    else
    {
      v33 = v7[1];
      sub_74A390(v33, 0, 0, v12, 0, &qword_4CF7CE0);
    }
    sub_74D110(v33, 0, 0, &qword_4CF7CE0);
    v13 = a1[1];
    if ( v13 && (*(_BYTE *)(v13 + 197) & 0x60) == 0x20 && (unsigned int)sub_8D3A70(v7[1]) && !sub_5D76E0() )
    {
      putc(38, stream);
      ++dword_4CF7F40;
    }
    if ( (unsigned int)sub_8D2E30(v7[1]) )
    {
      v14 = sub_8D46C0(v7[1]);
      if ( (unsigned int)sub_8D2310(v14) )
        sub_74F590(v7[1], 1, &qword_4CF7CE0);
    }
LABEL_13:
    v7 = (__int64 *)*v7;
    v10 = stream;
    if ( !v7 )
      break;
    putc(44, stream);
    v4 = stream;
    ++dword_4CF7F40;
    putc(32, stream);
    ++dword_4CF7F40;
  }
  if ( (a1[2] & 1) != 0 )
  {
    v19 = 44;
    v20 = " ...";
    while ( 1 )
    {
      putc(v19, v10);
      v19 = *v20++;
      if ( !(_BYTE)v19 )
        break;
      v10 = stream;
    }
    dword_4CF7F40 += 5;
LABEL_39:
    v10 = stream;
  }
LABEL_30:
  putc(41, v10);
  ++dword_4CF7F40;
  byte_4CF7D6F = v37;
  return sub_750520(&qword_4CF7CE0);
}
