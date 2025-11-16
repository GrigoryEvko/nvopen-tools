// Function: sub_897CB0
// Address: 0x897cb0
//
void __fastcall sub_897CB0(__int64 a1, __int64 a2)
{
  int v2; // edx
  _QWORD **v3; // rcx
  __int64 *v4; // rbx
  _QWORD *i; // rax
  unsigned int v6; // r10d
  size_t v7; // rax
  _QWORD *v8; // rax
  _QWORD *v9; // rsi
  __int64 v10; // r8
  char v11; // r12
  __m128i v12; // xmm1
  int v13; // r14d
  int v14; // eax
  __m128i v15; // xmm2
  __m128i v16; // xmm3
  int v17; // r12d
  __m128i v18; // xmm4
  __int64 v19; // rax
  unsigned int v21; // [rsp+18h] [rbp-168h]
  __int64 v22; // [rsp+18h] [rbp-168h]
  __int64 v23; // [rsp+20h] [rbp-160h]
  unsigned int v24; // [rsp+20h] [rbp-160h]
  int v25; // [rsp+28h] [rbp-158h]
  int v26; // [rsp+2Ch] [rbp-154h]
  _QWORD *v27; // [rsp+38h] [rbp-148h]
  _OWORD v28[4]; // [rsp+40h] [rbp-140h] BYREF
  __m128i v29[6]; // [rsp+80h] [rbp-100h] BYREF
  char s[8]; // [rsp+E0h] [rbp-A0h] BYREF
  __int64 v31; // [rsp+E8h] [rbp-98h]
  __int128 v32; // [rsp+F0h] [rbp-90h]
  __int128 v33; // [rsp+100h] [rbp-80h]
  __int128 v34; // [rsp+110h] [rbp-70h]
  __int128 v35; // [rsp+120h] [rbp-60h]
  __int128 v36; // [rsp+130h] [rbp-50h]
  int v37; // [rsp+140h] [rbp-40h]

  v2 = 1;
  v3 = *(_QWORD ***)(a1 + 192);
  v4 = *(__int64 **)(a2 + 368);
  for ( i = *v3; i; ++v2 )
  {
    v3 = (_QWORD **)i;
    i = (_QWORD *)*i;
  }
  v27 = v3;
  if ( v4 )
  {
    v6 = 1;
    v25 = v2 - 1;
    do
    {
      v10 = v4[3];
      v11 = *((_BYTE *)v4 + 32);
      v12 = _mm_loadu_si128(xmmword_4F06660);
      v13 = v25 + v6;
      v14 = *((_DWORD *)v4 + 4);
      v33 = 0;
      v34 = 0;
      v15 = _mm_loadu_si128(&xmmword_4F06660[1]);
      v16 = _mm_loadu_si128(&xmmword_4F06660[2]);
      v17 = v11 & 1;
      v26 = v14;
      v18 = _mm_loadu_si128(&xmmword_4F06660[3]);
      *(_QWORD *)s = 0x2D6F7475613CLL;
      memset(v29, 0, 0x58u);
      v31 = 0;
      v37 = 0;
      v32 = 0;
      v35 = 0;
      v36 = 0;
      v29[1].m128i_i64[0] = *(__int64 *)((char *)v4 + 36);
      v19 = *(__int64 *)((char *)v4 + 44);
      v28[0] = v12;
      v29[1].m128i_i64[1] = v19;
      v28[1] = v15;
      v28[2] = v16;
      v28[3] = v18;
      *((_QWORD *)&v28[0] + 1) = *(__int64 *)((char *)v4 + 36);
      if ( v6 <= 9 )
      {
        s[6] = v6 + 48;
      }
      else
      {
        v22 = v10;
        v24 = v6;
        sub_622470(v6, &s[6]);
        v10 = v22;
        v6 = v24;
      }
      v21 = v6;
      v23 = v10;
      v7 = strlen(s);
      *(_WORD *)&s[v7] = 62;
      sub_878540(s, v7 + 1, (__int64 *)v28);
      v8 = sub_897A40(v13, (__int64)v28, 0, v17, v23, a1, v29);
      *(_BYTE *)(v8[1] + 83LL) |= 0x40u;
      v9 = v27;
      v27 = v8;
      *(_DWORD *)(v8[1] + 56LL) = v26;
      v6 = v21 + 1;
      *(_BYTE *)(v8[8] + 161LL) |= 4u;
      *((_DWORD *)v8 + 15) = v13;
      *v9 = v8;
      v4[1] = (__int64)v8;
      v4 = (__int64 *)*v4;
    }
    while ( v4 );
  }
  if ( (*(_BYTE *)(a2 + 133) & 0x10) == 0 )
    sub_644060(a2);
}
