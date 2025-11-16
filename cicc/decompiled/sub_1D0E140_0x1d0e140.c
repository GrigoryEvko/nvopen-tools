// Function: sub_1D0E140
// Address: 0x1d0e140
//
__int64 __fastcall sub_1D0E140(__int64 *a1, __int64 *a2, _QWORD *a3, _DWORD *a4)
{
  _DWORD *v4; // r9
  __int64 v6; // rbx
  __int64 v7; // r12
  unsigned __int64 v8; // rax
  unsigned __int64 v9; // rcx
  __int64 *v10; // r8
  __int64 *v11; // r14
  bool v12; // cf
  unsigned __int64 v13; // rax
  __int64 v14; // rcx
  __int64 v15; // r15
  __int64 v16; // rcx
  __int64 v17; // rsi
  int v18; // edi
  __int64 v19; // r15
  __int64 v20; // rcx
  __int64 v21; // rsi
  int v22; // eax
  int v23; // ecx
  char v24; // si
  __int64 v25; // rcx
  __int64 v26; // rdx
  __int64 i; // r14
  unsigned __int64 v28; // rdi
  unsigned __int64 v29; // rdi
  __int64 v31; // r15
  __int64 v32; // rax
  _DWORD *v33; // [rsp+0h] [rbp-60h]
  _QWORD *v34; // [rsp+8h] [rbp-58h]
  __int64 v35; // [rsp+10h] [rbp-50h]
  __int64 *v36; // [rsp+18h] [rbp-48h]
  __int64 *v37; // [rsp+18h] [rbp-48h]
  __int64 v38; // [rsp+20h] [rbp-40h]
  __int64 v39; // [rsp+20h] [rbp-40h]
  __int64 v40; // [rsp+28h] [rbp-38h]

  v4 = a4;
  v6 = a1[1];
  v7 = *a1;
  v8 = 0xF0F0F0F0F0F0F0F1LL * ((v6 - *a1) >> 4);
  if ( v8 == 0x78787878787878LL )
    sub_4262D8((__int64)"vector::_M_realloc_insert");
  v9 = 1;
  v10 = a2;
  v11 = a2;
  if ( v8 )
    v9 = 0xF0F0F0F0F0F0F0F1LL * ((v6 - v7) >> 4);
  v12 = __CFADD__(v9, v8);
  v13 = v9 - 0xF0F0F0F0F0F0F0FLL * ((v6 - v7) >> 4);
  v14 = (__int64)a2 - v7;
  if ( v12 )
  {
    v31 = 0x7FFFFFFFFFFFFF80LL;
  }
  else
  {
    if ( !v13 )
    {
      v35 = 0;
      v15 = 272;
      v40 = 0;
      goto LABEL_7;
    }
    if ( v13 > 0x78787878787878LL )
      v13 = 0x78787878787878LL;
    v31 = 272 * v13;
  }
  v33 = v4;
  v34 = a3;
  v32 = sub_22077B0(v31);
  v14 = (__int64)a2 - v7;
  v10 = a2;
  a3 = v34;
  v4 = v33;
  v40 = v32;
  v35 = v32 + v31;
  v15 = v32 + 272;
LABEL_7:
  v16 = v40 + v14;
  if ( v16 )
  {
    v17 = *a3;
    v18 = *v4;
    *(_QWORD *)(v16 + 8) = 0;
    *(_QWORD *)(v16 + 112) = v16 + 128;
    LODWORD(v4) = 0;
    *(_QWORD *)v16 = v17;
    *(_QWORD *)(v16 + 32) = v16 + 48;
    *(_BYTE *)(v16 + 236) &= 0xFCu;
    *(_QWORD *)(v16 + 16) = 0;
    *(_QWORD *)(v16 + 24) = 0;
    *(_QWORD *)(v16 + 40) = 0x400000000LL;
    *(_QWORD *)(v16 + 120) = 0x400000000LL;
    *(_DWORD *)(v16 + 192) = v18;
    *(_QWORD *)(v16 + 196) = 0;
    *(_QWORD *)(v16 + 204) = 0;
    *(_QWORD *)(v16 + 212) = 0;
    *(_QWORD *)(v16 + 220) = 0;
    *(_WORD *)(v16 + 228) = 0;
    *(_DWORD *)(v16 + 232) = 0;
    *(_QWORD *)(v16 + 240) = 0;
    *(_QWORD *)(v16 + 248) = 0;
    *(_QWORD *)(v16 + 256) = 0;
    *(_QWORD *)(v16 + 264) = 0;
  }
  if ( v10 != (__int64 *)v7 )
  {
    v19 = v40;
    v20 = v7;
    while ( 1 )
    {
      if ( v19 )
      {
        *(_QWORD *)v19 = *(_QWORD *)v20;
        *(_QWORD *)(v19 + 8) = *(_QWORD *)(v20 + 8);
        *(_QWORD *)(v19 + 16) = *(_QWORD *)(v20 + 16);
        v21 = *(_QWORD *)(v20 + 24);
        *(_DWORD *)(v19 + 40) = 0;
        *(_QWORD *)(v19 + 24) = v21;
        *(_QWORD *)(v19 + 32) = v19 + 48;
        *(_DWORD *)(v19 + 44) = 4;
        if ( *(_DWORD *)(v20 + 40) )
        {
          v36 = v10;
          v38 = v20;
          sub_1D0B890(v19 + 32, v20 + 32, (__int64)a3, v20, (int)v10, (int)v4);
          v10 = v36;
          v20 = v38;
        }
        *(_DWORD *)(v19 + 120) = 0;
        *(_QWORD *)(v19 + 112) = v19 + 128;
        *(_DWORD *)(v19 + 124) = 4;
        if ( *(_DWORD *)(v20 + 120) )
        {
          v37 = v10;
          v39 = v20;
          sub_1D0B890(v19 + 112, v20 + 112, (__int64)a3, v20, (int)v10, (int)v4);
          v10 = v37;
          v20 = v39;
        }
        *(_DWORD *)(v19 + 192) = *(_DWORD *)(v20 + 192);
        *(_DWORD *)(v19 + 196) = *(_DWORD *)(v20 + 196);
        *(_DWORD *)(v19 + 200) = *(_DWORD *)(v20 + 200);
        *(_DWORD *)(v19 + 204) = *(_DWORD *)(v20 + 204);
        *(_DWORD *)(v19 + 208) = *(_DWORD *)(v20 + 208);
        *(_DWORD *)(v19 + 212) = *(_DWORD *)(v20 + 212);
        *(_DWORD *)(v19 + 216) = *(_DWORD *)(v20 + 216);
        *(_DWORD *)(v19 + 220) = *(_DWORD *)(v20 + 220);
        *(_WORD *)(v19 + 224) = *(_WORD *)(v20 + 224);
        *(_WORD *)(v19 + 226) = *(_WORD *)(v20 + 226);
        *(_WORD *)(v19 + 228) = *(_WORD *)(v20 + 228);
        *(_DWORD *)(v19 + 232) = *(_DWORD *)(v20 + 232);
        *(_BYTE *)(v19 + 236) = *(_BYTE *)(v20 + 236) & 3 | *(_BYTE *)(v19 + 236) & 0xFC;
        *(_DWORD *)(v19 + 240) = *(_DWORD *)(v20 + 240);
        *(_DWORD *)(v19 + 244) = *(_DWORD *)(v20 + 244);
        *(_DWORD *)(v19 + 248) = *(_DWORD *)(v20 + 248);
        *(_DWORD *)(v19 + 252) = *(_DWORD *)(v20 + 252);
        *(_QWORD *)(v19 + 256) = *(_QWORD *)(v20 + 256);
        *(_QWORD *)(v19 + 264) = *(_QWORD *)(v20 + 264);
      }
      v20 += 272;
      if ( v10 == (__int64 *)v20 )
        break;
      v19 += 272;
    }
    v15 = v19 + 544;
  }
  if ( v10 != (__int64 *)v6 )
  {
    do
    {
      v25 = *v11;
      v26 = *((unsigned int *)v11 + 10);
      *(_DWORD *)(v15 + 40) = 0;
      *(_DWORD *)(v15 + 44) = 4;
      *(_QWORD *)v15 = v25;
      *(_QWORD *)(v15 + 8) = v11[1];
      *(_QWORD *)(v15 + 16) = v11[2];
      *(_QWORD *)(v15 + 24) = v11[3];
      *(_QWORD *)(v15 + 32) = v15 + 48;
      if ( (_DWORD)v26 )
        sub_1D0B890(v15 + 32, (__int64)(v11 + 4), v26, v15 + 48, (int)v10, (int)v4);
      v22 = *((_DWORD *)v11 + 30);
      *(_DWORD *)(v15 + 120) = 0;
      *(_QWORD *)(v15 + 112) = v15 + 128;
      *(_DWORD *)(v15 + 124) = 4;
      if ( v22 )
        sub_1D0B890(v15 + 112, (__int64)(v11 + 14), v26, v15 + 128, (int)v10, (int)v4);
      v23 = *((_DWORD *)v11 + 48);
      v24 = *((_BYTE *)v11 + 236);
      v11 += 34;
      v15 += 272;
      *(_DWORD *)(v15 - 80) = v23;
      *(_DWORD *)(v15 - 76) = *((_DWORD *)v11 - 19);
      *(_DWORD *)(v15 - 72) = *((_DWORD *)v11 - 18);
      *(_DWORD *)(v15 - 68) = *((_DWORD *)v11 - 17);
      *(_DWORD *)(v15 - 64) = *((_DWORD *)v11 - 16);
      *(_DWORD *)(v15 - 60) = *((_DWORD *)v11 - 15);
      *(_DWORD *)(v15 - 56) = *((_DWORD *)v11 - 14);
      *(_DWORD *)(v15 - 52) = *((_DWORD *)v11 - 13);
      *(_WORD *)(v15 - 48) = *((_WORD *)v11 - 24);
      *(_WORD *)(v15 - 46) = *((_WORD *)v11 - 23);
      *(_WORD *)(v15 - 44) = *((_WORD *)v11 - 22);
      *(_DWORD *)(v15 - 40) = *((_DWORD *)v11 - 10);
      *(_BYTE *)(v15 - 36) = v24 & 3 | *(_BYTE *)(v15 - 36) & 0xFC;
      *(_DWORD *)(v15 - 32) = *((_DWORD *)v11 - 8);
      *(_DWORD *)(v15 - 28) = *((_DWORD *)v11 - 7);
      *(_DWORD *)(v15 - 24) = *((_DWORD *)v11 - 6);
      *(_DWORD *)(v15 - 20) = *((_DWORD *)v11 - 5);
      *(_QWORD *)(v15 - 16) = *(v11 - 2);
      *(_QWORD *)(v15 - 8) = *(v11 - 1);
    }
    while ( (__int64 *)v6 != v11 );
  }
  for ( i = v7; i != v6; i += 272 )
  {
    v28 = *(_QWORD *)(i + 112);
    if ( v28 != i + 128 )
      _libc_free(v28);
    v29 = *(_QWORD *)(i + 32);
    if ( v29 != i + 48 )
      _libc_free(v29);
  }
  if ( v7 )
    j_j___libc_free_0(v7, a1[2] - v7);
  a1[1] = v15;
  *a1 = v40;
  a1[2] = v35;
  return v35;
}
