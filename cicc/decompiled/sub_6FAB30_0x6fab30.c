// Function: sub_6FAB30
// Address: 0x6fab30
//
__int64 __fastcall sub_6FAB30(const __m128i *a1, __int64 a2, unsigned int a3, unsigned int a4, int a5)
{
  int v6; // ebx
  __int64 v7; // rdx
  __int64 v8; // rcx
  __int64 v9; // r8
  __int64 v10; // r9
  __int64 v11; // r13
  __int64 v12; // r9
  __int64 v13; // rcx
  __int64 i; // r8
  char v16; // dl
  __int64 v17; // rax
  __int64 v18; // rdx
  __int64 v19; // rax
  __int64 v20; // rax
  __int64 v21; // r13
  __int64 j; // r15
  __int64 v23; // rax
  __int64 v24; // rax
  __int64 v25; // rcx
  __int64 v26; // r9
  __int64 v27; // r8
  __int64 v28; // rax
  __int64 *v29; // r13
  _DWORD *v30; // r15
  __int64 v31; // rax
  __int64 v32; // [rsp-10h] [rbp-1C0h]
  __int64 v33; // [rsp-8h] [rbp-1B8h]
  __int64 v34; // [rsp+0h] [rbp-1B0h]
  __int64 v35; // [rsp+0h] [rbp-1B0h]
  __int64 *v38; // [rsp+18h] [rbp-198h] BYREF
  _OWORD v39[4]; // [rsp+20h] [rbp-190h] BYREF
  _OWORD v40[5]; // [rsp+60h] [rbp-150h] BYREF
  __m128i v41; // [rsp+B0h] [rbp-100h]
  __m128i v42; // [rsp+C0h] [rbp-F0h]
  __m128i v43; // [rsp+D0h] [rbp-E0h]
  __m128i v44; // [rsp+E0h] [rbp-D0h]
  __m128i v45; // [rsp+F0h] [rbp-C0h]
  __m128i v46; // [rsp+100h] [rbp-B0h]
  __m128i v47; // [rsp+110h] [rbp-A0h]
  __m128i v48; // [rsp+120h] [rbp-90h]
  __m128i v49; // [rsp+130h] [rbp-80h]
  __m128i v50; // [rsp+140h] [rbp-70h]
  __m128i v51; // [rsp+150h] [rbp-60h]
  __m128i v52; // [rsp+160h] [rbp-50h]
  __m128i v53; // [rsp+170h] [rbp-40h]

  v6 = sub_8D3110(a2);
  v11 = sub_8D46C0(a2);
  if ( v6 )
  {
    if ( a1[1].m128i_i8[1] == 2 )
      goto LABEL_3;
    sub_6ED0A0((__int64)a1);
  }
  if ( a1[1].m128i_i8[1] != 2 )
  {
    sub_6ECEA0((__int64)a1, &a1[4].m128i_i32[1], v7, v8, v9, v10);
    v13 = a1[1].m128i_u8[0];
    if ( !(_BYTE)v13 )
      return sub_6E6870((__int64)a1);
    goto LABEL_8;
  }
LABEL_3:
  sub_6FA350(a1, a2);
  v13 = a1[1].m128i_u8[0];
  if ( !(_BYTE)v13 )
    return sub_6E6870((__int64)a1);
LABEL_8:
  i = a1->m128i_i64[0];
  v16 = *(_BYTE *)(a1->m128i_i64[0] + 140);
  if ( v16 == 12 )
  {
    v17 = a1->m128i_i64[0];
    do
    {
      v17 = *(_QWORD *)(v17 + 160);
      v16 = *(_BYTE *)(v17 + 140);
    }
    while ( v16 == 12 );
  }
  if ( !v16 )
    return sub_6E6870((__int64)a1);
  v18 = *(unsigned __int8 *)(v11 + 140);
  if ( (_BYTE)v18 == 12 )
  {
    v19 = v11;
    do
    {
      v19 = *(_QWORD *)(v19 + 160);
      v18 = *(unsigned __int8 *)(v19 + 140);
    }
    while ( (_BYTE)v18 == 12 );
  }
  if ( (_BYTE)v18 )
  {
    v39[0] = _mm_loadu_si128(a1);
    v39[1] = _mm_loadu_si128(a1 + 1);
    v39[2] = _mm_loadu_si128(a1 + 2);
    v39[3] = _mm_loadu_si128(a1 + 3);
    v40[0] = _mm_loadu_si128(a1 + 4);
    v40[1] = _mm_loadu_si128(a1 + 5);
    v40[2] = _mm_loadu_si128(a1 + 6);
    v40[3] = _mm_loadu_si128(a1 + 7);
    v40[4] = _mm_loadu_si128(a1 + 8);
    if ( (_BYTE)v13 == 2 )
    {
      v41 = _mm_loadu_si128(a1 + 9);
      v42 = _mm_loadu_si128(a1 + 10);
      v43 = _mm_loadu_si128(a1 + 11);
      v44 = _mm_loadu_si128(a1 + 12);
      v45 = _mm_loadu_si128(a1 + 13);
      v46 = _mm_loadu_si128(a1 + 14);
      v47 = _mm_loadu_si128(a1 + 15);
      v48 = _mm_loadu_si128(a1 + 16);
      v49 = _mm_loadu_si128(a1 + 17);
      v50 = _mm_loadu_si128(a1 + 18);
      v51 = _mm_loadu_si128(a1 + 19);
      v52 = _mm_loadu_si128(a1 + 20);
      v53 = _mm_loadu_si128(a1 + 21);
    }
    else if ( (_BYTE)v13 == 5 || (_BYTE)v13 == 1 )
    {
      v41.m128i_i64[0] = a1[9].m128i_i64[0];
    }
    if ( !a5 )
    {
      v34 = i;
      if ( (unsigned int)sub_8D3A70(i) )
      {
        if ( (unsigned int)sub_8D3A70(v11) )
        {
          for ( i = v34; *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
            ;
          for ( j = v11; *(_BYTE *)(j + 140) == 12; j = *(_QWORD *)(j + 160) )
            ;
          if ( i != j )
          {
            v12 = dword_4F07588;
            if ( !dword_4F07588 || (v23 = *(_QWORD *)(i + 32), *(_QWORD *)(j + 32) != v23) || !v23 )
            {
              v35 = i;
              v24 = sub_8D5CE0(i, j);
              if ( v24 )
              {
                sub_6F7270(a1, v24, (_DWORD *)v11, a3, 0, a4, 0, 0);
                v27 = v33;
LABEL_38:
                if ( !v6 )
                {
                  if ( a1[1].m128i_i8[0] == 1 )
                    sub_6E84C0(a1[9].m128i_i64[0], a2);
                  return sub_6E4F10((__int64)a1, (__int64)v39, a4, a1[1].m128i_i8[1] == 1);
                }
                if ( !a4 )
                {
                  *(_BYTE *)(a1[9].m128i_i64[0] + 58) |= 0x80u;
                  v28 = sub_6F6F40(a1, 0, 0, v25, v27, v26);
                  v29 = (__int64 *)sub_73DC30(7, v11, v28);
                  sub_6E84C0((__int64)v29, a2);
                  sub_6E7150(v29, (__int64)a1);
                  goto LABEL_41;
                }
                v31 = sub_6F6F40(a1, 0, a4, v25, v27, v26);
                v21 = sub_73DC30(7, v11, v31);
                sub_6E84C0(v21, a2);
LABEL_22:
                *(_BYTE *)(v21 + 27) |= 2u;
LABEL_23:
                sub_6E7150((__int64 *)v21, (__int64)a1);
                if ( !v6 )
                  return sub_6E4F10((__int64)a1, (__int64)v39, a4, a1[1].m128i_i8[1] == 1);
LABEL_41:
                sub_6ED1A0((__int64)a1);
                return sub_6E4F10((__int64)a1, (__int64)v39, a4, a1[1].m128i_i8[1] == 1);
              }
              v30 = (_DWORD *)sub_8D5CE0(j, v35);
              if ( v30 )
              {
                v38 = (__int64 *)sub_6F6F40(a1, 0, v18, v13, i, v12);
                sub_6E7880(v11, v30, 1, 0, (__int64 *)&v38, (_DWORD *)v40 + 1, 0);
                sub_6E7150(v38, (__int64)a1);
                v25 = v32;
                goto LABEL_38;
              }
            }
          }
        }
      }
    }
    v20 = sub_6F6F40(a1, 0, v18, v13, i, v12);
    v21 = sub_73DC30(7, v11, v20);
    sub_6E84C0(v21, a2);
    if ( !a4 )
      goto LABEL_23;
    goto LABEL_22;
  }
  return sub_6E6840((__int64)a1);
}
