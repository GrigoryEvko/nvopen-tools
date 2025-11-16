// Function: sub_392BB70
// Address: 0x392bb70
//
__int64 __fastcall sub_392BB70(__int64 a1, _QWORD *a2)
{
  int v2; // eax
  __int64 v3; // rdx
  __int64 v4; // rax
  __int64 v5; // rax
  bool v6; // cf
  char v7; // si
  __int64 v8; // rcx
  __int64 v9; // rax
  __m128i v10; // xmm0
  __int64 v12; // rax
  __m128i si128; // xmm0
  unsigned __int64 v14; // [rsp+8h] [rbp-48h] BYREF
  unsigned __int64 v15[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v16[6]; // [rsp+20h] [rbp-30h] BYREF

  v2 = sub_392A7D0(a2);
  if ( v2 == 92 )
    v2 = sub_392A7D0(a2);
  if ( v2 == -1 )
  {
    v14 = 25;
    v15[0] = (unsigned __int64)v16;
    v12 = sub_22409D0((__int64)v15, &v14, 0);
    si128 = _mm_load_si128((const __m128i *)&xmmword_3F90150);
    v15[0] = v12;
    v16[0] = v14;
    *(_QWORD *)(v12 + 16) = 0x746F757120656C67LL;
    *(_BYTE *)(v12 + 24) = 101;
    *(__m128i *)v12 = si128;
  }
  else
  {
    if ( (unsigned int)sub_392A7D0(a2) == 39 )
    {
      v3 = a2[13];
      v4 = a2[18];
      v6 = v4 == v3;
      v5 = v4 - v3;
      if ( v6 || v5 == 1 || *(_WORD *)v3 != 23591 )
      {
        v8 = *(char *)(v3 + 1);
        goto LABEL_17;
      }
      v7 = *(_BYTE *)(v3 + 2);
      if ( v7 == 110 )
      {
        v8 = 10;
        goto LABEL_17;
      }
      if ( v7 > 110 )
      {
        v8 = 9;
        if ( v7 == 116 )
          goto LABEL_17;
      }
      else
      {
        v8 = 39;
        if ( v7 == 39 )
        {
LABEL_17:
          *(_QWORD *)(a1 + 16) = v5;
          *(_DWORD *)a1 = 4;
          *(_QWORD *)(a1 + 8) = v3;
          *(_DWORD *)(a1 + 32) = 64;
          *(_QWORD *)(a1 + 24) = v8;
          return a1;
        }
        if ( v7 == 98 )
        {
          v8 = 8;
          goto LABEL_17;
        }
      }
      v8 = v7;
      goto LABEL_17;
    }
    v14 = 25;
    v15[0] = (unsigned __int64)v16;
    v9 = sub_22409D0((__int64)v15, &v14, 0);
    v10 = _mm_load_si128((const __m128i *)&xmmword_3F90160);
    v15[0] = v9;
    v16[0] = v14;
    *(_QWORD *)(v9 + 16) = 0x6E6F6C206F6F7420LL;
    *(_BYTE *)(v9 + 24) = 103;
    *(__m128i *)v9 = v10;
  }
  v15[1] = v14;
  *(_BYTE *)(v15[0] + v14) = 0;
  sub_392A760(a1, a2, a2[13], v15);
  if ( (_QWORD *)v15[0] != v16 )
    j_j___libc_free_0(v15[0]);
  return a1;
}
