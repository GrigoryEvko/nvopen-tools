// Function: sub_64F620
// Address: 0x64f620
//
__int64 __fastcall sub_64F620(__int64 a1, __int64 a2, __int64 *a3, __int64 a4, __int64 a5, __int64 a6)
{
  int v6; // r15d
  __int64 v8; // rbx
  __int64 v9; // rsi
  __int64 v10; // rdi
  __int64 v11; // rdx
  __int64 v12; // rcx
  int v13; // r13d
  char v14; // r14
  int v15; // r12d
  __int64 v16; // rsi
  const char *v17; // rdi
  size_t v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // r15
  __int64 v21; // rdx
  __int64 v22; // rcx
  __int64 v23; // r8
  __int64 v24; // rax
  char v25; // dl
  __int64 v27; // rcx
  int v28; // eax
  __int64 v29; // [rsp+0h] [rbp-E0h]
  __int64 v30; // [rsp+8h] [rbp-D8h]
  unsigned int v31; // [rsp+14h] [rbp-CCh]
  __int64 v32; // [rsp+18h] [rbp-C8h]
  char v33; // [rsp+18h] [rbp-C8h]
  char v34; // [rsp+18h] [rbp-C8h]
  __int16 v35; // [rsp+22h] [rbp-BEh]
  int v36; // [rsp+24h] [rbp-BCh]
  int v37; // [rsp+28h] [rbp-B8h]
  int v38; // [rsp+2Ch] [rbp-B4h]
  int v39; // [rsp+38h] [rbp-A8h] BYREF
  int v40; // [rsp+3Ch] [rbp-A4h] BYREF
  __int64 v41; // [rsp+40h] [rbp-A0h] BYREF
  __int64 v42; // [rsp+48h] [rbp-98h] BYREF
  _BYTE v43[144]; // [rsp+50h] [rbp-90h] BYREF

  v6 = a1;
  LODWORD(v8) = 0;
  v38 = a2;
  v39 = 0;
  v41 = sub_724DC0(a1, a2, a3, a4, a5, a6);
  v40 = 0;
  if ( dword_4F04C58 != -1 && (_DWORD)a2 )
    LODWORD(v8) = (*(_BYTE *)(*(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 216) + 198LL) & 0x10) != 0;
  if ( (_DWORD)a1 )
  {
    sub_854AB0();
    v9 = 8;
    v10 = (__int64)a3;
    sub_6446A0(a3, 8u);
  }
  else
  {
    v9 = (__int64)dword_4F07508;
    v10 = 412;
    sub_6851C0(412, dword_4F07508);
    sub_854B40();
  }
  v31 = dword_4F063F8;
  v35 = unk_4F063FC;
  if ( word_4F06418[0] == 137 )
  {
    if ( (_DWORD)v8 )
      sub_6851C0(3516, dword_4F07508);
    LOBYTE(v13) = 0;
    v14 = 0;
    v15 = 0;
    v16 = 2;
    sub_724C70(v41, 2);
    v8 = v41;
    v17 = (const char *)unk_4F061F0;
    *(_QWORD *)(v41 + 184) = unk_4F061F0;
    v18 = strlen(v17) + 1;
    *(_QWORD *)(v8 + 176) = v18;
    *(_QWORD *)(v8 + 128) = sub_73CA60(v18);
    LOBYTE(v8) = 0;
    unk_4F061D8 = qword_4F063F0;
    sub_7B8B50(v18, 2, qword_4F063F0, v19);
    v29 = 0;
    v30 = 0;
    v32 = 0;
    LOBYTE(v36) = 0;
    LOBYTE(v37) = 0;
    if ( v6 )
      goto LABEL_10;
LABEL_46:
    v20 = 0;
    sub_724E30(&v41);
    return v20;
  }
  sub_7B8B50(v10, v9, v11, v12);
  v36 = 0;
  v37 = 0;
  LOBYTE(v13) = BYTE4(qword_4F077B4) | v8;
  if ( HIDWORD(qword_4F077B4) | (unsigned int)v8 )
  {
    v13 = 0;
    while ( 1 )
    {
      while ( 1 )
      {
        v42 = *(_QWORD *)&dword_4F063F8;
        if ( (unsigned __int16)(word_4F06418[0] - 81) <= 0x26u )
          break;
        if ( (unsigned __int16)(word_4F06418[0] - 263) <= 3u )
          goto LABEL_22;
LABEL_31:
        if ( qword_4F077A8 <= 0x9E33u )
          goto LABEL_39;
        if ( word_4F06418[0] == 91 )
        {
          if ( v36 )
            goto LABEL_36;
          sub_7B8B50(v10, v9, qword_4F077A8, &qword_4F077A8);
          v36 = 1;
        }
        else
        {
          if ( word_4F06418[0] != 154 || qword_4F077A8 <= 0x11363u )
            goto LABEL_39;
          if ( v37 )
          {
LABEL_36:
            sub_7B8B50(v10, v9, qword_4F077A8, &qword_4F077A8);
            goto LABEL_37;
          }
          sub_7B8B50(v10, v9, qword_4F077A8, &qword_4F077A8);
          v37 = 1;
        }
      }
      v27 = 0x6004000001LL;
      if ( !_bittest64(&v27, (unsigned int)word_4F06418[0] - 81) )
        goto LABEL_31;
LABEL_22:
      v9 = 0;
      v10 = (__int64)v43;
      v28 = sub_624060((__int64)v43);
      if ( (v28 & 0xFFFFFFFC) != 0 )
      {
        v9 = (__int64)&v42;
        v10 = 1289;
        v33 = v28;
        sub_6851C0(1289, &v42);
        LOBYTE(v28) = v33;
      }
      else if ( (v28 & 1) != 0 )
      {
        v9 = (__int64)&v42;
        v10 = 1287;
        v34 = v28;
        sub_684B30(1287, &v42);
        LOBYTE(v28) = v34;
      }
      if ( (v28 & 2) != 0 )
      {
        if ( unk_4D04320 )
        {
          v9 = (__int64)&v42;
          v10 = 1612;
          sub_684B30(1612, &v42);
        }
        if ( v13 )
        {
LABEL_37:
          v9 = (__int64)&v42;
          v10 = 3264;
          sub_6851C0(3264, &v42);
        }
        else
        {
          v13 = 1;
        }
      }
    }
  }
LABEL_39:
  sub_7BE280(27, 125, 0, 0);
  ++*(_BYTE *)(qword_4F061C8 + 36LL);
  if ( word_4F06418[0] == 7 )
  {
    if ( !HIDWORD(qword_4F077B4) || (unk_4F063A8 & 7) == 0 )
    {
      sub_72A510(&unk_4F06300, v41);
      sub_7BDAB0(&v40);
      goto LABEL_42;
    }
    sub_6851D0(2479);
  }
  else
  {
    sub_6851D0(194);
  }
  sub_72C970(v41);
LABEL_42:
  if ( HIDWORD(qword_4F077B4) | (unsigned int)v8 && v38 && word_4F06418[0] == 55 )
  {
    v32 = sub_704000(&v40, &v39);
    if ( v39 == -1 )
      v39 = 0;
    v30 = sub_7053C0(&v40);
    if ( v36 )
    {
      v29 = sub_705620(&v40);
    }
    else
    {
      v29 = 0;
      if ( word_4F06418[0] != 28 )
        sub_6851D0(18);
    }
    if ( v32 )
    {
      v14 = 1;
      v15 = 1;
      if ( (*(_BYTE *)(v32 + 24) & 2) != 0 )
        v14 = v13;
    }
    else
    {
      v14 = 1;
      v15 = 1;
    }
  }
  else
  {
    v29 = 0;
    v14 = 1;
    v15 = 0;
    v30 = 0;
    v32 = 0;
  }
  sub_7BE280(28, 18, 0, 0);
  v16 = 65;
  --*(_BYTE *)(qword_4F061C8 + 36LL);
  unk_4F061D8 = qword_4F063F0;
  sub_7BE280(75, 65, 0, 0);
  if ( !v6 )
    goto LABEL_46;
LABEL_10:
  v20 = sub_726270();
  v24 = sub_724E50(&v41, v16, v21, v22, v23);
  v25 = *(_BYTE *)(v20 + 128);
  *(_QWORD *)(v20 + 120) = v24;
  *(_DWORD *)(v20 + 64) = v31;
  *(_WORD *)(v20 + 68) = v35;
  *(_BYTE *)(v20 + 128) = v25 & 0xC0 | ((32 * v36) | (16 * v37) | (8 * v13) | (4 * v14) | v8 | (2 * v15)) & 0x3F;
  *(_QWORD *)(v20 + 136) = v32;
  *(_QWORD *)(v20 + 144) = v30;
  *(_QWORD *)(v20 + 152) = v29;
  *(_QWORD *)(v20 + 160) = v39;
  if ( v15 )
  {
    sub_703CC0(v20);
    if ( v38 )
      return v20;
  }
  else if ( v38 )
  {
    return v20;
  }
  sub_733510(v20);
  if ( !dword_4F04C3C )
    sub_8699D0(v20, 43, 0);
  return v20;
}
