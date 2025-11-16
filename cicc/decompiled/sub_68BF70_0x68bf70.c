// Function: sub_68BF70
// Address: 0x68bf70
//
__int64 __fastcall sub_68BF70(const __m128i *a1, _QWORD *a2, _DWORD *a3, __int64 a4, _QWORD *a5)
{
  __int64 v8; // rax
  __int64 v9; // r13
  _DWORD *v10; // r12
  __int64 v12; // rbx
  __int64 i; // rdi
  __int64 v14; // rax
  __int64 v15; // rax
  int v16; // edi
  __int8 v17; // al
  _BOOL8 v18; // rsi
  __int64 v19; // r8
  __int64 v20; // r9
  __int64 j; // rdx
  __int64 v22; // rax
  __int64 v23; // [rsp-8h] [rbp-198h]
  __m128i v24; // [rsp+0h] [rbp-190h]
  __m128i v25; // [rsp+10h] [rbp-180h]
  __m128i v26; // [rsp+20h] [rbp-170h]
  __m128i v27; // [rsp+30h] [rbp-160h]
  _OWORD v28[5]; // [rsp+40h] [rbp-150h] BYREF
  __m128i v29; // [rsp+90h] [rbp-100h]
  __m128i v30; // [rsp+A0h] [rbp-F0h]
  __m128i v31; // [rsp+B0h] [rbp-E0h]
  __m128i v32; // [rsp+C0h] [rbp-D0h]
  __m128i v33; // [rsp+D0h] [rbp-C0h]
  __m128i v34; // [rsp+E0h] [rbp-B0h]
  __m128i v35; // [rsp+F0h] [rbp-A0h]
  __m128i v36; // [rsp+100h] [rbp-90h]
  __m128i v37; // [rsp+110h] [rbp-80h]
  __m128i v38; // [rsp+120h] [rbp-70h]
  __m128i v39; // [rsp+130h] [rbp-60h]
  __m128i v40; // [rsp+140h] [rbp-50h]
  __m128i v41; // [rsp+150h] [rbp-40h]

  v8 = sub_6EB5C0();
  *a5 = 0;
  v9 = v8;
  if ( a2 )
  {
    if ( *a2 )
    {
      v10 = (_DWORD *)sub_6E1A20(*a2);
      if ( (unsigned int)sub_6E5430() )
        sub_6851C0(0x8Cu, v10);
    }
    else
    {
      sub_6E65B0(a2);
      v12 = a2[3];
      if ( *(_BYTE *)(v12 + 25) == 1 )
        sub_6FA3A0(v12 + 8);
      for ( i = *(_QWORD *)(v12 + 8); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
        ;
      v14 = sub_732700(i, i, 0, 0, 0, 0, 0, 0);
      v15 = sub_68A000(v9, v14);
      v9 = *(_QWORD *)(v15 + 88);
      v16 = v15;
      v24 = _mm_loadu_si128(a1);
      v25 = _mm_loadu_si128(a1 + 1);
      v26 = _mm_loadu_si128(a1 + 2);
      v17 = a1[1].m128i_i8[0];
      v27 = _mm_loadu_si128(a1 + 3);
      v28[0] = _mm_loadu_si128(a1 + 4);
      v28[1] = _mm_loadu_si128(a1 + 5);
      v28[2] = _mm_loadu_si128(a1 + 6);
      v28[3] = _mm_loadu_si128(a1 + 7);
      v28[4] = _mm_loadu_si128(a1 + 8);
      if ( v17 == 2 )
      {
        v29 = _mm_loadu_si128(a1 + 9);
        v30 = _mm_loadu_si128(a1 + 10);
        v31 = _mm_loadu_si128(a1 + 11);
        v32 = _mm_loadu_si128(a1 + 12);
        v33 = _mm_loadu_si128(a1 + 13);
        v34 = _mm_loadu_si128(a1 + 14);
        v35 = _mm_loadu_si128(a1 + 15);
        v36 = _mm_loadu_si128(a1 + 16);
        v37 = _mm_loadu_si128(a1 + 17);
        v38 = _mm_loadu_si128(a1 + 18);
        v39 = _mm_loadu_si128(a1 + 19);
        v40 = _mm_loadu_si128(a1 + 20);
        v41 = _mm_loadu_si128(a1 + 21);
      }
      else if ( v17 == 5 || v17 == 1 )
      {
        v29.m128i_i64[0] = a1[9].m128i_i64[0];
      }
      v18 = (a1[1].m128i_i8[2] & 0x40) != 0;
      sub_6EAB60(v16, v18, 0, (unsigned int)v28 + 4, (unsigned int)v28 + 12, a1[5].m128i_i64[1], (__int64)a1);
      j = v23;
      if ( a1[1].m128i_i8[0] )
      {
        v22 = a1->m128i_i64[0];
        for ( j = *(unsigned __int8 *)(a1->m128i_i64[0] + 140); (_BYTE)j == 12; j = *(unsigned __int8 *)(v22 + 140) )
          v22 = *(_QWORD *)(v22 + 160);
        if ( (_BYTE)j )
        {
          v18 = 0;
          sub_6F5FA0(
            a1,
            0,
            0,
            1,
            v19,
            v20,
            v24.m128i_i64[0],
            v24.m128i_i64[1],
            v25.m128i_i64[0],
            v25.m128i_i64[1],
            v26.m128i_i64[0],
            v26.m128i_i64[1],
            v27.m128i_i64[0],
            v27.m128i_i64[1],
            *(_QWORD *)&v28[0],
            *((_QWORD *)&v28[0] + 1));
        }
      }
      *a5 = sub_6F7150(v12 + 8, v18, j);
    }
  }
  else if ( (unsigned int)sub_6E5430() )
  {
    sub_6851C0(0xA5u, a3);
  }
  return v9;
}
