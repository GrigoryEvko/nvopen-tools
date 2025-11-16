// Function: sub_1095CD0
// Address: 0x1095cd0
//
__int64 __fastcall sub_1095CD0(__int64 a1, __int64 a2)
{
  char *v3; // rsi
  char v4; // dl
  char *v5; // rax
  _BYTE *v6; // rcx
  __int64 v7; // rax
  char v9; // al
  _BYTE *v10; // rax
  __m128i *v11; // rax
  __int64 v12; // rdx
  __int64 v13; // [rsp+8h] [rbp-48h] BYREF
  _QWORD v14[2]; // [rsp+10h] [rbp-40h] BYREF
  _QWORD v15[6]; // [rsp+20h] [rbp-30h] BYREF

  v3 = *(char **)(a2 + 152);
  v4 = *v3;
  if ( (unsigned __int8)(*v3 - 48) <= 9u )
  {
    v5 = v3 + 1;
    do
    {
      *(_QWORD *)(a2 + 152) = v5;
      v4 = *v5;
      v3 = v5++;
    }
    while ( (unsigned __int8)(v4 - 48) <= 9u );
  }
  if ( ((v4 - 43) & 0xFD) != 0 )
  {
    if ( (v4 & 0xDF) == 0x45 )
    {
      v6 = v3 + 1;
      *(_QWORD *)(a2 + 152) = v3 + 1;
      v9 = v3[1];
      if ( ((v9 - 43) & 0xFD) == 0 )
      {
        v6 = v3 + 2;
        *(_QWORD *)(a2 + 152) = v3 + 2;
        v9 = v3[2];
      }
      if ( (unsigned __int8)(v9 - 48) <= 9u )
      {
        v10 = v6 + 1;
        do
        {
          v6 = v10;
          *(_QWORD *)(a2 + 152) = v10++;
        }
        while ( (unsigned __int8)(*v6 - 48) <= 9u );
      }
    }
    else
    {
      v6 = *(_BYTE **)(a2 + 152);
    }
    v7 = *(_QWORD *)(a2 + 104);
    *(_DWORD *)a1 = 6;
    *(_DWORD *)(a1 + 32) = 64;
    *(_QWORD *)(a1 + 8) = v7;
    *(_QWORD *)(a1 + 16) = &v6[-v7];
    *(_QWORD *)(a1 + 24) = 0;
    return a1;
  }
  v13 = 29;
  v14[0] = v15;
  v11 = (__m128i *)sub_22409D0(v14, &v13, 0);
  v14[0] = v11;
  v15[0] = v13;
  *v11 = _mm_load_si128((const __m128i *)&xmmword_3F90090);
  v12 = v14[0];
  qmemcpy(&v11[1], "float literal", 13);
  v14[1] = v13;
  *(_BYTE *)(v12 + v13) = 0;
  sub_1095C00(a1, a2, *(_QWORD *)(a2 + 152), (__int64)v14);
  if ( (_QWORD *)v14[0] == v15 )
    return a1;
  j_j___libc_free_0(v14[0], v15[0] + 1LL);
  return a1;
}
