// Function: sub_10978A0
// Address: 0x10978a0
//
__int64 __fastcall sub_10978A0(__int64 a1, __int64 a2)
{
  int v2; // eax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rax
  bool v6; // cf
  __int64 v7; // rcx
  __int64 v9; // rax
  __m128i si128; // xmm0
  __m128i *v11; // rax
  __m128i v12; // xmm0
  __int64 v13; // rdx
  __int64 v14; // rax
  __int64 v15; // rax
  __m128i v16; // xmm0
  __int64 v17; // rax
  __m128i v18; // xmm0
  __int64 v19; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v20[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v21[6]; // [rsp+20h] [rbp-30h] BYREF

  v2 = sub_1095C70((_QWORD *)a2);
  if ( *(_BYTE *)(a2 + 129) )
  {
    v19 = 35;
    v20[0] = v21;
    v9 = sub_22409D0(v20, &v19, 0);
    v20[0] = v9;
    v21[0] = v19;
    *(__m128i *)v9 = _mm_load_si128((const __m128i *)&xmmword_3F90130);
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F90140);
    *(_WORD *)(v9 + 32) = 27745;
    *(_BYTE *)(v9 + 34) = 115;
    *(__m128i *)(v9 + 16) = si128;
  }
  else
  {
    if ( *(_BYTE *)(a2 + 118) )
    {
      while ( 1 )
      {
        if ( v2 == -1 )
        {
          v19 = 28;
          v20[0] = v21;
          v11 = (__m128i *)sub_22409D0(v20, &v19, 0);
          v12 = _mm_load_si128((const __m128i *)&xmmword_3F90170);
          v20[0] = v11;
          v21[0] = v19;
          qmemcpy(&v11[1], "ing constant", 12);
          *v11 = v12;
          goto LABEL_18;
        }
        if ( v2 == 39 )
        {
          if ( (unsigned int)sub_1095CA0((_QWORD *)a2) != 39 )
          {
            v13 = *(_QWORD *)(a2 + 104);
            v14 = *(_QWORD *)(a2 + 152);
            *(_DWORD *)a1 = 3;
            *(_DWORD *)(a1 + 32) = 64;
            *(_QWORD *)(a1 + 8) = v13;
            *(_QWORD *)(a1 + 16) = v14 - v13;
            *(_QWORD *)(a1 + 24) = 0;
            return a1;
          }
          sub_1095C70((_QWORD *)a2);
        }
        v2 = sub_1095C70((_QWORD *)a2);
      }
    }
    if ( v2 == 92 )
      v2 = sub_1095C70((_QWORD *)a2);
    if ( v2 == -1 )
    {
      v19 = 25;
      v20[0] = v21;
      v17 = sub_22409D0(v20, &v19, 0);
      v18 = _mm_load_si128((const __m128i *)&xmmword_3F90150);
      v20[0] = v17;
      v21[0] = v19;
      *(_QWORD *)(v17 + 16) = 0x746F757120656C67LL;
      *(_BYTE *)(v17 + 24) = 101;
      *(__m128i *)v17 = v18;
    }
    else
    {
      if ( (unsigned int)sub_1095C70((_QWORD *)a2) == 39 )
      {
        v3 = *(_QWORD *)(a2 + 104);
        v4 = *(_QWORD *)(a2 + 152);
        v6 = v4 == v3;
        v5 = v4 - v3;
        if ( !v6 && v5 != 1 && *(_WORD *)v3 == 23591 )
        {
          v7 = *(char *)(v3 + 2);
          if ( (_BYTE)v7 == 39 )
          {
            v7 = 39;
          }
          else
          {
            switch ( (char)v7 )
            {
              case 'b':
                v7 = 8;
                break;
              case 'f':
                v7 = 12;
                break;
              case 'n':
                v7 = 10;
                break;
              case 'r':
                v7 = 13;
                break;
              case 't':
                v7 = 9;
                break;
              default:
                break;
            }
          }
        }
        else
        {
          v7 = *(char *)(v3 + 1);
        }
        *(_QWORD *)(a1 + 16) = v5;
        *(_DWORD *)a1 = 4;
        *(_QWORD *)(a1 + 8) = v3;
        *(_DWORD *)(a1 + 32) = 64;
        *(_QWORD *)(a1 + 24) = v7;
        return a1;
      }
      v19 = 25;
      v20[0] = v21;
      v15 = sub_22409D0(v20, &v19, 0);
      v16 = _mm_load_si128((const __m128i *)&xmmword_3F90160);
      v20[0] = v15;
      v21[0] = v19;
      *(_QWORD *)(v15 + 16) = 0x6E6F6C206F6F7420LL;
      *(_BYTE *)(v15 + 24) = 103;
      *(__m128i *)v15 = v16;
    }
  }
LABEL_18:
  v20[1] = v19;
  *(_BYTE *)(v20[0] + v19) = 0;
  sub_1095C00(a1, a2, *(_QWORD *)(a2 + 104), (__int64)v20);
  if ( (_QWORD *)v20[0] != v21 )
    j_j___libc_free_0(v20[0], v21[0] + 1LL);
  return a1;
}
