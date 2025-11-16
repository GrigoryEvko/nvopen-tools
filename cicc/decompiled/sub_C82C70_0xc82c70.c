// Function: sub_C82C70
// Address: 0xc82c70
//
__int64 __fastcall sub_C82C70(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5)
{
  DIR *v6; // rdi
  _BYTE *v7; // rdx
  __m128i v8; // xmm1
  __m128i v9; // xmm2
  __m128i v10; // xmm3
  __int64 v11; // rdx
  __int64 v13; // [rsp+10h] [rbp-60h] BYREF
  __int128 v14; // [rsp+20h] [rbp-50h] BYREF
  __int128 v15; // [rsp+30h] [rbp-40h] BYREF
  __int128 v16; // [rsp+40h] [rbp-30h] BYREF
  int v17; // [rsp+50h] [rbp-20h]
  int v18; // [rsp+54h] [rbp-1Ch]

  v6 = *(DIR **)a1;
  if ( v6 )
    closedir(v6);
  v7 = *(_BYTE **)(a1 + 8);
  v14 = 0;
  v15 = 0;
  v16 = 0;
  v17 = 0;
  LODWORD(v14) = 9;
  BYTE4(v14) = 1;
  v18 = 0xFFFF;
  *(_QWORD *)a1 = 0;
  *(_QWORD *)(a1 + 16) = 0;
  *v7 = 0;
  LOBYTE(v13) = 0;
  v8 = _mm_loadu_si128((const __m128i *)((char *)&v14 + 8));
  v9 = _mm_loadu_si128((const __m128i *)((char *)&v15 + 8));
  v10 = _mm_loadu_si128((const __m128i *)((char *)&v16 + 8));
  *(_DWORD *)(a1 + 40) = v14;
  v11 = BYTE4(v14);
  *(__m128i *)(a1 + 48) = v8;
  *(__m128i *)(a1 + 64) = v9;
  *(_BYTE *)(a1 + 44) = v11;
  *(__m128i *)(a1 + 80) = v10;
  sub_2241E40(&v13, a2, v11, a4, a5);
  return 0;
}
