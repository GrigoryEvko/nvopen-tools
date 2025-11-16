// Function: sub_25F1790
// Address: 0x25f1790
//
__int64 __fastcall sub_25F1790(__int64 a1, __int64 *a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  __int64 v6; // rax
  __int64 v7; // rdx
  __int64 v8; // rdx
  __int64 v9; // rdx
  __int64 *v10; // rdx
  __int64 v11; // rdx
  char v12; // al
  __int64 v13; // rdx
  __int64 result; // rax
  __int64 v15; // rdx
  unsigned int v16; // r13d
  __int64 *v17; // rdx
  size_t v18; // rdx
  int v19; // edx

  v6 = *a2;
  *(_QWORD *)(a1 + 56) = 1;
  *(_QWORD *)a1 = v6;
  *(_BYTE *)(a1 + 8) = *((_BYTE *)a2 + 8);
  *(_QWORD *)(a1 + 16) = a2[2];
  *(_QWORD *)(a1 + 24) = a2[3];
  *(_QWORD *)(a1 + 32) = a2[4];
  *(_QWORD *)(a1 + 40) = a2[5];
  LOBYTE(v6) = *((_BYTE *)a2 + 48);
  *(_QWORD *)(a1 + 64) = 0;
  *(_QWORD *)(a1 + 72) = 0;
  *(_DWORD *)(a1 + 80) = 0;
  v7 = a2[8];
  *(_BYTE *)(a1 + 48) = v6;
  LODWORD(v6) = *((_DWORD *)a2 + 20);
  *(_QWORD *)(a1 + 64) = v7;
  v8 = a2[9];
  ++a2[7];
  *(_QWORD *)(a1 + 72) = v8;
  *(_DWORD *)(a1 + 80) = v6;
  a2[8] = 0;
  a2[9] = 0;
  *((_DWORD *)a2 + 20) = 0;
  *(_QWORD *)(a1 + 88) = a1 + 104;
  *(_QWORD *)(a1 + 96) = 0;
  v9 = *((unsigned int *)a2 + 24);
  if ( (_DWORD)v9 )
    sub_25EFD80(a1 + 88, (char **)a2 + 11, v9, a4, a5, a6);
  *(_QWORD *)(a1 + 104) = a1 + 120;
  *(_QWORD *)(a1 + 112) = 0x600000000LL;
  if ( *((_DWORD *)a2 + 28) )
    sub_25EFD80(a1 + 104, (char **)a2 + 13, v9, a4, a5, a6);
  *(_QWORD *)(a1 + 168) = a1 + 184;
  v10 = (__int64 *)a2[21];
  if ( v10 == a2 + 23 )
  {
    *(__m128i *)(a1 + 184) = _mm_loadu_si128((const __m128i *)(a2 + 23));
  }
  else
  {
    *(_QWORD *)(a1 + 168) = v10;
    *(_QWORD *)(a1 + 184) = a2[23];
  }
  v11 = a2[22];
  a2[21] = (__int64)(a2 + 23);
  a2[22] = 0;
  *(_QWORD *)(a1 + 176) = v11;
  v12 = *((_BYTE *)a2 + 200);
  *((_BYTE *)a2 + 184) = 0;
  *(_QWORD *)(a1 + 216) = 0;
  *(_QWORD *)(a1 + 224) = 0;
  *(_DWORD *)(a1 + 232) = 0;
  v13 = a2[27];
  *(_BYTE *)(a1 + 200) = v12;
  result = *((unsigned int *)a2 + 58);
  *(_QWORD *)(a1 + 216) = v13;
  v15 = a2[28];
  ++a2[26];
  a2[27] = 0;
  a2[28] = 0;
  *((_DWORD *)a2 + 58) = 0;
  *(_QWORD *)(a1 + 208) = 1;
  *(_QWORD *)(a1 + 224) = v15;
  *(_DWORD *)(a1 + 232) = result;
  *(_QWORD *)(a1 + 240) = a1 + 256;
  *(_QWORD *)(a1 + 248) = 0;
  v16 = *((_DWORD *)a2 + 62);
  if ( v16 )
  {
    result = (__int64)(a2 + 30);
    if ( (__int64 *)(a1 + 240) != a2 + 30 )
    {
      v17 = (__int64 *)a2[30];
      result = (__int64)(a2 + 32);
      if ( v17 == a2 + 32 )
      {
        result = sub_C8D5F0(a1 + 240, (const void *)(a1 + 256), v16, 8u, a5, a6);
        v18 = 8LL * *((unsigned int *)a2 + 62);
        if ( v18 )
          result = (__int64)memcpy(*(void **)(a1 + 240), (const void *)a2[30], v18);
        *(_DWORD *)(a1 + 248) = v16;
        *((_DWORD *)a2 + 62) = 0;
      }
      else
      {
        *(_QWORD *)(a1 + 240) = v17;
        v19 = *((_DWORD *)a2 + 63);
        *(_DWORD *)(a1 + 248) = v16;
        *(_DWORD *)(a1 + 252) = v19;
        a2[30] = result;
        a2[31] = 0;
      }
    }
  }
  return result;
}
