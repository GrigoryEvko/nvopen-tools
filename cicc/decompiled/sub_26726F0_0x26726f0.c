// Function: sub_26726F0
// Address: 0x26726f0
//
_DWORD *__fastcall sub_26726F0(_DWORD *a1, __int64 a2, __int64 a3)
{
  unsigned __int64 v4; // rbx
  int v5; // r10d
  __int64 v6; // rsi
  unsigned int v7; // eax
  __int64 *v8; // r13
  __int64 v9; // rdx
  unsigned __int64 v10; // rbx
  __int64 v11; // rcx
  __int64 v12; // r8
  __int64 v13; // r9
  int v14; // r10d
  unsigned int v15; // eax
  __int64 v16; // rcx
  __int64 *v17; // r13
  __int64 v18; // rdx
  __int64 v19; // rcx
  __int64 v20; // r8
  __int64 v21; // r9
  int v23; // r8d
  __int64 v24; // [rsp+8h] [rbp-148h]
  unsigned __int64 v25[38]; // [rsp+20h] [rbp-130h] BYREF

  v4 = a3 & 0xFFFFFFFFFFFFFFFBLL;
  v5 = *(_DWORD *)(a2 + 280);
  v6 = *(_QWORD *)(a2 + 264);
  if ( v5 )
  {
    v7 = (v5 - 1) & (v4 ^ (v4 >> 9));
    v8 = (__int64 *)(v6 + ((unsigned __int64)v7 << 7));
    v9 = *v8;
    if ( v4 == *v8 )
    {
LABEL_3:
      v10 = v4 | 4;
      v24 = a2;
      LODWORD(v25[0]) = *((_DWORD *)v8 + 2);
      sub_C8CD80((__int64)&v25[1], (__int64)&v25[5], (__int64)(v8 + 2), a2, (__int64)&v25[1], (__int64)&v25[5]);
      sub_C8CD80((__int64)&v25[7], (__int64)&v25[11], (__int64)(v8 + 8), v11, v12, v13);
      v5 = *(_DWORD *)(v24 + 280);
      v6 = *(_QWORD *)(v24 + 264);
      if ( !v5 )
        goto LABEL_16;
      goto LABEL_4;
    }
    v23 = 1;
    while ( v9 != -4 )
    {
      v7 = (v5 - 1) & (v23 + v7);
      v8 = (__int64 *)(v6 + ((unsigned __int64)v7 << 7));
      v9 = *v8;
      if ( v4 == *v8 )
        goto LABEL_3;
      ++v23;
    }
  }
  v10 = v4 | 4;
  memset(v25, 0, 0x78u);
  LODWORD(v25[0]) = 65793;
  v25[2] = (unsigned __int64)&v25[5];
  LODWORD(v25[3]) = 2;
  BYTE4(v25[4]) = 1;
  v25[8] = (unsigned __int64)&v25[11];
  LODWORD(v25[9]) = 4;
  BYTE4(v25[10]) = 1;
  if ( !v5 )
    goto LABEL_16;
LABEL_4:
  v14 = v5 - 1;
  v15 = v14 & (v10 ^ (v10 >> 9));
  v16 = (unsigned __int64)v15 << 7;
  v17 = (__int64 *)(v6 + v16);
  v18 = *(_QWORD *)(v6 + v16);
  if ( v18 == v10 )
  {
LABEL_5:
    LODWORD(v25[16]) = *((_DWORD *)v17 + 2);
    sub_C8CD80((__int64)&v25[17], (__int64)&v25[21], (__int64)(v17 + 2), v16, (__int64)&v25[1], (__int64)&v25[5]);
    sub_C8CD80((__int64)&v25[23], (__int64)&v25[27], (__int64)(v17 + 8), v19, v20, v21);
    goto LABEL_6;
  }
  v16 = 1;
  while ( v18 != -4 )
  {
    v15 = v14 & (v16 + v15);
    v17 = (__int64 *)(v6 + ((unsigned __int64)v15 << 7));
    v18 = *v17;
    if ( v10 == *v17 )
      goto LABEL_5;
    v16 = (unsigned int)(v16 + 1);
  }
LABEL_16:
  memset(&v25[16], 0, 0x78u);
  v25[24] = (unsigned __int64)&v25[27];
  LODWORD(v25[16]) = 65793;
  v25[18] = (unsigned __int64)&v25[21];
  LODWORD(v25[19]) = 2;
  BYTE4(v25[20]) = 1;
  LODWORD(v25[25]) = 4;
  BYTE4(v25[26]) = 1;
LABEL_6:
  *a1 = v25[0];
  sub_C8CF70((__int64)(a1 + 2), a1 + 10, 2, (__int64)&v25[5], (__int64)&v25[1]);
  sub_C8CF70((__int64)(a1 + 14), a1 + 22, 4, (__int64)&v25[11], (__int64)&v25[7]);
  a1[30] = v25[16];
  sub_C8CF70((__int64)(a1 + 32), a1 + 40, 2, (__int64)&v25[21], (__int64)&v25[17]);
  sub_C8CF70((__int64)(a1 + 44), a1 + 52, 4, (__int64)&v25[27], (__int64)&v25[23]);
  if ( !BYTE4(v25[26]) )
    _libc_free(v25[24]);
  if ( !BYTE4(v25[20]) )
    _libc_free(v25[18]);
  if ( !BYTE4(v25[10]) )
  {
    _libc_free(v25[8]);
    if ( BYTE4(v25[4]) )
      return a1;
LABEL_18:
    _libc_free(v25[2]);
    return a1;
  }
  if ( !BYTE4(v25[4]) )
    goto LABEL_18;
  return a1;
}
