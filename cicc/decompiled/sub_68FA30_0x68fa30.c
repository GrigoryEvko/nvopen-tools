// Function: sub_68FA30
// Address: 0x68fa30
//
__int64 __fastcall sub_68FA30(__int64 a1, _DWORD *a2, const __m128i *a3, __int64 a4)
{
  __int64 result; // rax
  __int64 v7; // rax
  __int64 v8; // rax
  __int64 v9; // r12
  __int64 v10; // rax
  __int64 v11; // rax
  __int8 v12; // al
  __int64 v13; // r15
  __int64 v14; // rax
  _BOOL4 v15; // [rsp+Ch] [rbp-194h] BYREF
  __m128i v16[9]; // [rsp+10h] [rbp-190h] BYREF
  __m128i v17; // [rsp+A0h] [rbp-100h]
  __m128i v18; // [rsp+B0h] [rbp-F0h]
  __m128i v19; // [rsp+C0h] [rbp-E0h]
  __m128i v20; // [rsp+D0h] [rbp-D0h]
  __m128i v21; // [rsp+E0h] [rbp-C0h]
  __m128i v22; // [rsp+F0h] [rbp-B0h]
  __m128i v23; // [rsp+100h] [rbp-A0h]
  __m128i v24; // [rsp+110h] [rbp-90h]
  __m128i v25; // [rsp+120h] [rbp-80h]
  __m128i v26; // [rsp+130h] [rbp-70h]
  __m128i v27; // [rsp+140h] [rbp-60h]
  __m128i v28; // [rsp+150h] [rbp-50h]
  __m128i v29; // [rsp+160h] [rbp-40h]

  sub_6F69D0(a3, 8);
  result = sub_6F68A0(a3, a4);
  if ( !(_DWORD)result )
  {
    if ( unk_4D04950
      && (unsigned int)sub_8D2E30(a1)
      && (v7 = sub_8D46C0(a1), (unsigned int)sub_8D2310(v7))
      && (unsigned int)sub_8D2E30(a3->m128i_i64[0])
      && (v8 = sub_8D46C0(a3->m128i_i64[0]), (unsigned int)sub_8D2310(v8)) )
    {
      sub_6E5C80(unk_4F07470, 379, a2);
      if ( (a3[1].m128i_i8[2] & 4) != 0 )
      {
        v15 = (*(_BYTE *)(a4 + 18) & 2) != 0;
        sub_68F8E0(v16, a3);
        v9 = sub_6F6F40(a3, 0);
        sub_6FFCF0(a4, &v15);
        *(_QWORD *)(v9 + 16) = sub_6F6F40(a4, 0);
        v10 = sub_73DBF0(102, a3->m128i_i64[0], v9);
        sub_6E70E0(v10, a3);
        result = sub_6E4BC0(a3, v16);
      }
      else
      {
        result = sub_6E59E0(a4);
      }
      a3[1].m128i_i8[2] &= ~1u;
    }
    else if ( dword_4F077BC
           && qword_4F077A8 > 0x9DCFu
           && ((unsigned int)sub_8D4C80(a1)
            || (unsigned int)sub_8D2E30(a1) && (v11 = sub_8D46C0(a1), (unsigned int)sub_8D2310(v11)))
           && (unsigned int)sub_8D3D10(a3->m128i_i64[0]) )
    {
      v16[0] = _mm_loadu_si128(a3);
      v12 = a3[1].m128i_i8[0];
      v16[1] = _mm_loadu_si128(a3 + 1);
      v16[2] = _mm_loadu_si128(a3 + 2);
      v16[3] = _mm_loadu_si128(a3 + 3);
      v16[4] = _mm_loadu_si128(a3 + 4);
      v16[5] = _mm_loadu_si128(a3 + 5);
      v16[6] = _mm_loadu_si128(a3 + 6);
      v16[7] = _mm_loadu_si128(a3 + 7);
      v16[8] = _mm_loadu_si128(a3 + 8);
      if ( v12 == 2 )
      {
        v17 = _mm_loadu_si128(a3 + 9);
        v18 = _mm_loadu_si128(a3 + 10);
        v19 = _mm_loadu_si128(a3 + 11);
        v20 = _mm_loadu_si128(a3 + 12);
        v21 = _mm_loadu_si128(a3 + 13);
        v22 = _mm_loadu_si128(a3 + 14);
        v23 = _mm_loadu_si128(a3 + 15);
        v24 = _mm_loadu_si128(a3 + 16);
        v25 = _mm_loadu_si128(a3 + 17);
        v26 = _mm_loadu_si128(a3 + 18);
        v27 = _mm_loadu_si128(a3 + 19);
        v28 = _mm_loadu_si128(a3 + 20);
        v29 = _mm_loadu_si128(a3 + 21);
      }
      else if ( v12 == 5 || v12 == 1 )
      {
        v17.m128i_i64[0] = a3[9].m128i_i64[0];
      }
      v13 = sub_6F6F40(a4, 0);
      *(_QWORD *)(v13 + 16) = sub_6F6F40(a3, 0);
      v14 = sub_73DBF0((unsigned int)((*(_BYTE *)(a4 + 18) & 2) != 0) + 98, a1, v13);
      sub_6E70E0(v14, a3);
      sub_6E4BC0(a3, v16);
      a3[1].m128i_i8[2] &= ~1u;
      if ( (*(_BYTE *)(a4 + 20) & 1) != 0 )
      {
        result = sub_6E53E0(5, 2380, a2);
        if ( (_DWORD)result )
          return sub_684B30(0x94Cu, a2);
      }
      else
      {
        result = sub_6E53E0(5, 2220, a2);
        if ( (_DWORD)result )
          return sub_684B30(0x8ACu, a2);
      }
    }
    else
    {
      result = sub_6E68E0(300, a3);
      a3[1].m128i_i8[2] &= ~1u;
    }
  }
  return result;
}
