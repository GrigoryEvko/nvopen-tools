// Function: sub_625150
// Address: 0x625150
//
__int64 __fastcall sub_625150(__int64 a1, __int64 a2, _BYTE *a3)
{
  _BYTE *v4; // r9
  __int16 v6; // r15
  __int64 v7; // rsi
  char v8; // cl
  char v9; // cl
  __int64 *v10; // r13
  char v11; // r12
  char v12; // r12
  __int64 v13; // rdi
  unsigned int *v14; // rsi
  __int64 v15; // rdi
  __int64 v16; // rdx
  __int64 v17; // rcx
  _BYTE *v19; // [rsp+10h] [rbp-220h]
  unsigned int v20; // [rsp+1Ch] [rbp-214h]
  _QWORD v21[66]; // [rsp+20h] [rbp-210h] BYREF

  v4 = a3;
  v6 = unk_4F061DC;
  v7 = qword_4F04C68[0] + 776LL * dword_4F04C64;
  v20 = unk_4F061D8;
  memset(v21, 0, 0x1D8u);
  v21[19] = v21;
  v21[3] = *(_QWORD *)&dword_4F063F8;
  if ( dword_4F077BC && qword_4F077A8 <= 0x9F5Fu )
    BYTE2(v21[22]) |= 1u;
  v8 = *(_BYTE *)(a1 + 89);
  v21[0] = *(_QWORD *)a1;
  v9 = v8 & 4;
  if ( (*(_BYTE *)(a1 + 195) & 8) == 0 )
  {
    v12 = 0;
    v10 = 0;
LABEL_9:
    v21[36] = *(_QWORD *)(a1 + 152);
    if ( !v9 )
      goto LABEL_10;
    goto LABEL_20;
  }
  if ( !v9 )
  {
    v10 = *(__int64 **)(a1 + 248);
    if ( !v10 )
    {
      v12 = 0;
      v21[36] = *(_QWORD *)(a1 + 152);
      goto LABEL_10;
    }
    v10 = (__int64 *)*v10;
    v11 = *((_BYTE *)v10 + 83);
    *((_BYTE *)v10 + 83) = v11 | 0x40;
    v12 = (v11 & 0x40) != 0;
    v9 = *(_BYTE *)(a1 + 89) & 4;
    goto LABEL_9;
  }
  v12 = 0;
  v10 = 0;
  v21[36] = *(_QWORD *)(a1 + 152);
LABEL_20:
  if ( *(_QWORD *)(*(_QWORD *)(**(_QWORD **)(*(_QWORD *)(a1 + 40) + 32LL) + 96LL) + 56LL)
    || (*(_BYTE *)(a1 + 195) & 0xB) == 1 )
  {
    BYTE1(v21[16]) |= 0x40u;
    *(_QWORD *)(v7 + 624) = v21;
    *(_BYTE *)(v7 + 11) |= 0x40u;
    if ( a3 )
      goto LABEL_11;
    goto LABEL_22;
  }
  BYTE1(v21[16]) |= 0x80u;
LABEL_10:
  *(_BYTE *)(v7 + 11) |= 0x40u;
  *(_QWORD *)(v7 + 624) = v21;
  if ( a3 )
    goto LABEL_11;
LABEL_22:
  v4 = *(_BYTE **)(*(_QWORD *)(*(_QWORD *)(a1 + 152) + 168LL) + 56LL);
LABEL_11:
  v13 = a2;
  v19 = v4;
  sub_7BC160(a2);
  if ( dword_4F077C4 == 2 )
    *(_BYTE *)(qword_4F04C68[0] + 776LL * (int)dword_4F04C40 + 7) |= 8u;
  if ( (*v19 & 1) == 0 )
    sub_721090(v13);
  v14 = 0;
  sub_623870((unsigned __int64)v19, 0, (__int64)v21, 0);
  v15 = a1;
  sub_884800(a1);
  if ( dword_4F077C4 == 2 )
  {
    v15 = (int)dword_4F04C40;
    v16 = 776LL * (int)dword_4F04C40;
    *(_BYTE *)(qword_4F04C68[0] + v16 + 7) &= ~8u;
    v17 = qword_4F04C68[0];
    if ( *(_QWORD *)(qword_4F04C68[0] + v16 + 456) )
      sub_8845B0(v15);
  }
  if ( v10 )
  {
    v16 = *((_BYTE *)v10 + 83) & 0xBF;
    *((_BYTE *)v10 + 83) = *((_BYTE *)v10 + 83) & 0xBF | (v12 << 6);
  }
  if ( word_4F06418[0] != 9 )
  {
    v14 = &dword_4F063F8;
    v15 = 18;
    sub_6851C0(18, &dword_4F063F8);
    while ( word_4F06418[0] != 9 )
      sub_7B8B50(18, &dword_4F063F8, v16, v17);
  }
  sub_7B8B50(v15, v14, v16, v17);
  unk_4F061DC = v6;
  unk_4F061D8 = v20;
  return v20;
}
