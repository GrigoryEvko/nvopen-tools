// Function: sub_73A460
// Address: 0x73a460
//
__int64 __fastcall sub_73A460(const __m128i *a1, __int64 a2, __int64 a3, __int64 a4, _UNKNOWN *__ptr32 *a5)
{
  __int64 v6; // rcx
  __int64 v7; // r8
  __int64 v8; // r9
  __m128i *v9; // r14
  __m128i *v10; // r13
  _DWORD *v12; // rdx
  __int64 v13; // r15
  unsigned int v14; // eax
  _UNKNOWN *__ptr32 *v15; // r8
  unsigned __int64 v16; // rcx
  __m128i **v17; // rbx
  char v18; // di
  char v19; // r15
  bool v20; // zf
  __m128i *v21; // rax
  unsigned int v22; // eax

  if ( dword_4F07588 && (unsigned int)sub_73A390((__int64)a1, a2, a3, dword_4F07588, a5) )
  {
    v9 = (__m128i *)a1->m128i_i64[0];
    if ( a1->m128i_i64[0] )
      return v9[5].m128i_i64[1];
    v12 = dword_4F07270;
    if ( dword_4F07270[0] == unk_4F073B8 || !(unsigned int)sub_72AA80((__int64)a1) )
    {
      v19 = 0;
      v22 = sub_72DB90((__int64)a1, a2, (__int64)v12, v6, v7, v8);
      v16 = (18957679 * (unsigned __int64)v22) >> 32;
      v17 = (__m128i **)(qword_4F07AE0 + 8LL * (v22 % 0x7F7));
      if ( unk_4F068AC )
        goto LABEL_24;
      v18 = a1[10].m128i_i8[13];
      if ( v18 == 2 )
      {
        v17 = 0;
        goto LABEL_37;
      }
      if ( v18 != 6 )
        goto LABEL_24;
    }
    else
    {
      if ( dword_4F04C58 == -1 )
      {
        v18 = a1[10].m128i_i8[13];
        v17 = 0;
        goto LABEL_32;
      }
      v13 = *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 256);
      if ( !v13 )
      {
        v13 = sub_85E950();
        *(_QWORD *)(qword_4F04C68[0] + 776LL * dword_4F04C58 + 256) = v13;
      }
      v14 = sub_72DB90((__int64)a1, a2, (__int64)v12, v6, v7, v8);
      v16 = 31 * (v14 / 0x1F);
      v17 = (__m128i **)(v13 + 8LL * (v14 % 0x1F + 1));
      if ( unk_4F068AC )
        goto LABEL_15;
      v18 = a1[10].m128i_i8[13];
      if ( v18 == 2 )
      {
        v17 = 0;
        goto LABEL_32;
      }
      if ( v18 != 6 )
      {
LABEL_15:
        v19 = 1;
        goto LABEL_16;
      }
      v19 = 1;
    }
    if ( a1[11].m128i_i8[0] == 2 && *(_BYTE *)(a1[11].m128i_i64[1] + 173) == 2 )
    {
      v18 = 6;
      v17 = 0;
LABEL_31:
      if ( v19 )
      {
LABEL_32:
        v10 = (__m128i *)sub_724D50(v18);
LABEL_33:
        sub_72A510(a1, v10);
        sub_73B910(v10);
        if ( !v17 )
          return (__int64)v10;
        v21 = *v17;
LABEL_35:
        v10[7].m128i_i64[1] = (__int64)v21;
        *v17 = v10;
        return (__int64)v10;
      }
LABEL_37:
      v10 = (__m128i *)sub_724D80(v18);
      goto LABEL_33;
    }
LABEL_24:
    if ( !v17 )
    {
LABEL_25:
      v18 = a1[10].m128i_i8[13];
      goto LABEL_31;
    }
LABEL_16:
    v10 = *v17;
    if ( *v17 )
    {
      while ( 1 )
      {
        v20 = (unsigned int)sub_739430((__int64)v10, (__int64)a1, 1u, v16, v15) == 0;
        v21 = (__m128i *)v10[7].m128i_i64[1];
        if ( !v20 )
          break;
        v9 = v10;
        if ( !v21 )
          goto LABEL_25;
        v10 = (__m128i *)v10[7].m128i_i64[1];
      }
      if ( v9 )
      {
        v9[7].m128i_i64[1] = (__int64)v21;
        v21 = *v17;
      }
      else
      {
        *v17 = v21;
      }
      goto LABEL_35;
    }
    goto LABEL_25;
  }
  return sub_740630(a1);
}
