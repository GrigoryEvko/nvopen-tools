// Function: sub_CC3C60
// Address: 0xcc3c60
//
__int64 __fastcall sub_CC3C60(
        int **a1,
        __int64 a2,
        __int64 a3,
        const __m128i *a4,
        __int64 a5,
        char a6,
        int a7,
        unsigned __int8 a8,
        char a9,
        __int64 a10)
{
  __int64 result; // rax
  __m128i *v12; // r12
  __int64 v14; // r13
  int *v15; // r14
  unsigned int *v16; // rax
  unsigned __int8 v17; // r8
  __int64 v18; // r12
  unsigned int *v19; // rbx
  unsigned int *v20; // rax
  __int64 v21; // rcx
  int v22; // edx
  char v23; // al
  char v24; // ecx^2
  int v25; // edx
  char v26; // al
  char v27; // ecx^2
  int v28; // edx
  char v29; // al
  char v30; // ecx^2
  int v31; // edx
  char v32; // al
  char v33; // ecx^2
  int v34; // edx
  char v35; // al
  char v36; // ecx^2
  int v37; // edx
  char v38; // al
  char v39; // ecx^2
  int v40; // edx
  char v41; // ecx^2
  int v42; // edx
  __int64 v45; // [rsp+18h] [rbp-78h]
  __m128i v46; // [rsp+20h] [rbp-70h]
  _DWORD v49[20]; // [rsp+40h] [rbp-50h] BYREF

  v45 = a2;
  result = a7 | (unsigned int)a8;
  if ( a2 )
  {
    v12 = (__m128i *)v49;
    do
    {
      v14 = a3;
      v15 = *a1;
      v46 = _mm_loadu_si128(a4 + 1);
      *v12 = _mm_loadu_si128(a4);
      v12[1] = v46;
      if ( a3 )
      {
        v16 = (unsigned int *)v12;
        v17 = a7 | a8;
        v18 = a5;
        v19 = v16;
        while ( v14 != 1 )
        {
          sub_CC2400(v19, v15, 0x40u, v18, v17);
          v17 = a7;
          v15 += 16;
          --v14;
        }
        v20 = v19;
        a5 = v18;
        v21 = v18;
        v12 = (__m128i *)v20;
        sub_CC2400(v20, v15, 0x40u, v21, a9 | v17);
      }
      v22 = v49[0];
      *(_BYTE *)a10 = v49[0];
      v23 = BYTE1(v22);
      v24 = BYTE2(v22);
      *(_BYTE *)(a10 + 3) = HIBYTE(v22);
      v25 = v49[1];
      *(_BYTE *)(a10 + 1) = v23;
      *(_BYTE *)(a10 + 2) = v24;
      v26 = BYTE1(v25);
      v27 = BYTE2(v25);
      *(_BYTE *)(a10 + 4) = v25;
      *(_BYTE *)(a10 + 7) = HIBYTE(v25);
      v28 = v49[2];
      *(_BYTE *)(a10 + 5) = v26;
      *(_BYTE *)(a10 + 6) = v27;
      v29 = BYTE1(v28);
      v30 = BYTE2(v28);
      *(_BYTE *)(a10 + 8) = v28;
      *(_BYTE *)(a10 + 11) = HIBYTE(v28);
      v31 = v49[3];
      *(_BYTE *)(a10 + 9) = v29;
      *(_BYTE *)(a10 + 10) = v30;
      v32 = BYTE1(v31);
      v33 = BYTE2(v31);
      *(_BYTE *)(a10 + 12) = v31;
      *(_BYTE *)(a10 + 15) = HIBYTE(v31);
      v34 = v49[4];
      *(_BYTE *)(a10 + 13) = v32;
      *(_BYTE *)(a10 + 14) = v33;
      v35 = BYTE1(v34);
      v36 = BYTE2(v34);
      *(_BYTE *)(a10 + 16) = v34;
      *(_BYTE *)(a10 + 19) = HIBYTE(v34);
      v37 = v49[5];
      *(_BYTE *)(a10 + 17) = v35;
      *(_BYTE *)(a10 + 18) = v36;
      v38 = BYTE1(v37);
      v39 = BYTE2(v37);
      *(_BYTE *)(a10 + 20) = v37;
      *(_BYTE *)(a10 + 23) = HIBYTE(v37);
      v40 = v49[6];
      *(_BYTE *)(a10 + 21) = v38;
      *(_BYTE *)(a10 + 22) = v39;
      v41 = BYTE2(v40);
      *(_WORD *)(a10 + 24) = v40;
      *(_BYTE *)(a10 + 27) = HIBYTE(v40);
      v42 = v49[7];
      *(_BYTE *)(a10 + 26) = v41;
      *(_BYTE *)(a10 + 28) = v42;
      result = BYTE1(v42);
      *(_BYTE *)(a10 + 29) = BYTE1(v42);
      a5 -= (a6 == 0) - 1LL;
      *(_BYTE *)(a10 + 30) = BYTE2(v42);
      a10 += 32;
      *(_BYTE *)(a10 - 1) = HIBYTE(v42);
      ++a1;
      --v45;
    }
    while ( v45 );
  }
  return result;
}
