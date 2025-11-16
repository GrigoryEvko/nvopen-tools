// Function: sub_A149E0
// Address: 0xa149e0
//
__int64 *__fastcall sub_A149E0(__int64 *a1, __int64 *a2, unsigned int a3, __int64 a4, __int64 a5)
{
  __int64 v5; // r15
  __int64 *v6; // r13
  __int64 v8; // rbx
  __int64 v9; // rsi
  __int64 v10; // rax
  unsigned __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // r12
  __int64 v14; // r12
  __int64 v15; // rax
  __m128i si128; // xmm0
  __m128i v17; // xmm0
  unsigned __int64 v19; // r14
  __int64 v20; // r14
  __int64 v21; // r14
  __int64 v22; // [rsp+8h] [rbp-78h]
  unsigned int v23; // [rsp+14h] [rbp-6Ch]
  unsigned int v24; // [rsp+18h] [rbp-68h]
  int v25; // [rsp+18h] [rbp-68h]
  __int64 v26; // [rsp+28h] [rbp-58h] BYREF
  _QWORD v27[2]; // [rsp+30h] [rbp-50h] BYREF
  _QWORD v28[8]; // [rsp+40h] [rbp-40h] BYREF

  v5 = a4;
  v6 = a1;
  v8 = a3;
  v9 = a2[1];
  v10 = *a2;
  v11 = (v9 - *a2) >> 5;
  if ( (_DWORD)v8 == (_DWORD)v11 )
  {
    v27[0] = a4;
    LODWORD(v26) = a5;
    if ( v9 == a2[2] )
    {
      sub_9C9BE0(a2, v9, v27, &v26);
    }
    else
    {
      if ( v9 )
      {
        *(_QWORD *)v9 = 6;
        *(_QWORD *)(v9 + 8) = 0;
        *(_QWORD *)(v9 + 16) = a4;
        if ( a4 != -4096 && a4 != 0 && a4 != -8192 )
          sub_BD73F0(v9);
        *(_DWORD *)(v9 + 24) = v26;
        v9 = a2[1];
      }
      a2[1] = v9 + 32;
    }
    goto LABEL_25;
  }
  if ( (unsigned int)v8 >= (unsigned int)v11 )
  {
    v19 = (unsigned int)(v8 + 1);
    if ( v19 > v11 )
    {
      a1 = a2;
      v24 = a5;
      v9 = v19 - v11;
      sub_9C9970(a2, v19 - v11);
      v10 = *a2;
      a5 = v24;
    }
    else if ( v19 < v11 )
    {
      v20 = 32 * v19;
      a4 = v10 + v20;
      v22 = v10 + v20;
      if ( v9 != v10 + v20 )
      {
        v21 = v10 + v20;
        do
        {
          v11 = *(_QWORD *)(v21 + 16);
          LOBYTE(a1) = v11 != -4096;
          LOBYTE(a4) = v11 != 0;
          if ( ((v11 != 0) & (unsigned __int8)a1) != 0 && v11 != -8192 )
          {
            a1 = (__int64 *)v21;
            v23 = a5;
            sub_BD60C0(v21);
            a5 = v23;
          }
          v21 += 32;
        }
        while ( v9 != v21 );
        a2[1] = v22;
        v10 = *a2;
      }
    }
  }
  v12 = v10 + 32 * v8;
  v13 = *(_QWORD *)(v12 + 16);
  if ( v13 )
  {
    if ( *(_QWORD *)(v5 + 8) != *(_QWORD *)(v13 + 8) )
    {
      v14 = sub_2241E50(a1, v9, v11, a4, a5);
      v26 = 57;
      v27[0] = v28;
      v15 = sub_22409D0(v27, &v26, 0);
      v27[0] = v15;
      v28[0] = v26;
      *(__m128i *)v15 = _mm_load_si128((const __m128i *)&xmmword_3F229B0);
      si128 = _mm_load_si128((const __m128i *)&xmmword_3F229C0);
      *(_QWORD *)(v15 + 48) = 0x6F69746172616C63LL;
      *(__m128i *)(v15 + 16) = si128;
      v17 = _mm_load_si128((const __m128i *)&xmmword_3F229D0);
      *(_BYTE *)(v15 + 56) = 110;
      *(__m128i *)(v15 + 32) = v17;
      v27[1] = v26;
      *(_BYTE *)(v27[0] + v26) = 0;
      sub_C63F00(v6, v27, 84, v14);
      if ( (_QWORD *)v27[0] != v28 )
        j_j___libc_free_0(v27[0], v28[0] + 1LL);
      return v6;
    }
    sub_BD84D0(*(_QWORD *)(v12 + 16), v5);
    sub_BD72D0(v13);
LABEL_25:
    *v6 = 1;
    return v6;
  }
  if ( v5 )
  {
    *(_QWORD *)(v12 + 16) = v5;
    if ( v5 != -8192 && v5 != -4096 )
    {
      v25 = a5;
      sub_BD73F0(v12);
      LODWORD(a5) = v25;
    }
  }
  *(_DWORD *)(v12 + 24) = a5;
  *v6 = 1;
  return v6;
}
