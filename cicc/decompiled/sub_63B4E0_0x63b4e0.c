// Function: sub_63B4E0
// Address: 0x63b4e0
//
__int64 __fastcall sub_63B4E0(__int64 a1, __int64 a2, __int64 a3)
{
  __int64 v3; // r13
  __int64 v4; // rax
  __int64 v5; // rcx
  __int64 v6; // rax
  unsigned __int8 v7; // dl
  unsigned int v8; // r15d
  __int64 v9; // rax
  unsigned int v10; // r14d
  _BYTE *v11; // rax
  unsigned int v13; // r14d
  __int64 v14; // rcx
  __int64 v15; // rax
  __int64 v16; // rcx
  __int64 v17; // [rsp+8h] [rbp-1B8h]
  __int64 v18; // [rsp+10h] [rbp-1B0h]
  char v19; // [rsp+18h] [rbp-1A8h]
  char v20; // [rsp+1Eh] [rbp-1A2h]
  bool v21; // [rsp+1Fh] [rbp-1A1h]
  _WORD v22[176]; // [rsp+20h] [rbp-1A0h] BYREF
  int v23; // [rsp+180h] [rbp-40h]
  __int16 v24; // [rsp+184h] [rbp-3Ch]

  v3 = sub_7ADF90(a1, a2, a3);
  v4 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  if ( !v4 )
LABEL_28:
    BUG();
  v5 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  while ( *(_BYTE *)(v4 + 4) != 6 )
  {
    v6 = *(int *)(v4 + 552);
    if ( (_DWORD)v6 != -1 )
    {
      v4 = qword_4F04C68[0] + 776 * v6;
      if ( v4 )
        continue;
    }
    goto LABEL_28;
  }
  v17 = 0;
  v19 = 0;
  v20 = *(_BYTE *)(a1 + 80);
  v18 = *(_QWORD *)(v4 + 208);
  if ( v20 == 8 )
  {
    v7 = *(_BYTE *)(v5 + 11);
    v19 = v7 >> 7;
    *(_BYTE *)(v5 + 11) = v7 | 0x80;
    v17 = qword_4CFDE58;
    qword_4CFDE58 = *(_QWORD *)(a1 + 88);
  }
  sub_7ADF70(v3, 1);
  memset(v22, 0, sizeof(v22));
  v22[37] = 257;
  v8 = dword_4F06650[0];
  HIBYTE(v22[33]) = 1;
  v9 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v23 = 0;
  v24 = 0;
  v21 = (*(_BYTE *)(v9 + 12) & 4) != 0;
  *(_BYTE *)(v9 + 12) |= 4u;
  sub_7BDB60(1);
  sub_7C6890(0, v22);
  v10 = dword_4F06650[0];
  sub_7AE700(unk_4F061C0 + 24LL, v8, dword_4F06650[0], 0, v3);
  sub_7AE340(v3);
  sub_7AE210(v3);
  sub_7BDC00();
  v11 = (_BYTE *)(qword_4F04C68[0] + 776LL * dword_4F04C64);
  v11[12] = (4 * v21) | v11[12] & 0xFB;
  if ( v20 == 8 )
  {
    v11[11] = (v19 << 7) | v11[11] & 0x7F;
    qword_4CFDE58 = v17;
  }
  if ( (dword_4F04C44 != -1 || (v11[6] & 2) != 0) && (*(_BYTE *)(v18 + 177) & 0x90) == 0x90 )
  {
    v13 = v10 - 1;
    if ( v20 == 8 || v20 == 21 )
    {
      v16 = v13;
      if ( v8 >= v13 )
        v16 = v8;
      *(_BYTE *)(sub_888280(a1, 0, v8, v16) + 50) = *(_QWORD *)(v3 + 8) == 0;
      if ( v20 == 8 )
        *(_QWORD *)(*(_QWORD *)(a1 + 104) + 8LL) = v3;
    }
    else if ( dword_4F077BC )
    {
      v14 = v13;
      if ( v8 >= v13 )
        v14 = v8;
      *(_BYTE *)(sub_888280(a1, 0, v8, v14) + 50) = *(_QWORD *)(v3 + 8) == 0;
      v15 = *(_QWORD *)(a1 + 104);
      if ( !v15 )
        v15 = sub_8790A0(a1, 0);
      *(_QWORD *)(v15 + 8) = v3;
    }
  }
  return v3;
}
