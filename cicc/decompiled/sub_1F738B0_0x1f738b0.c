// Function: sub_1F738B0
// Address: 0x1f738b0
//
__int64 __fastcall sub_1F738B0(__int64 a1, __int64 a2, unsigned int a3, unsigned int a4, int a5, char a6)
{
  __int64 v10; // rbx
  char v11; // di
  __int64 v12; // rax
  unsigned int v13; // eax
  __int16 v14; // dx
  char v15; // di
  __int64 v16; // rax
  unsigned int v17; // eax
  __int64 v19; // rax
  __int64 v20; // rbx
  unsigned int v21; // eax
  _QWORD *v22; // rsi
  __int64 v23; // rdx
  int v24; // eax
  __int64 v25; // rdx
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rax
  unsigned int v28; // r15d
  char v29; // al
  __m128i v30; // xmm0
  __int64 v31; // [rsp+8h] [rbp-78h]
  __m128i v32; // [rsp+10h] [rbp-70h] BYREF
  char v33; // [rsp+20h] [rbp-60h]
  __m128i v34; // [rsp+30h] [rbp-50h] BYREF
  char v35; // [rsp+40h] [rbp-40h]

  if ( a5 == 10 || !a6 && !sub_1D18C00(a2, 1, a3) )
    goto LABEL_19;
  v10 = *(_QWORD *)(a2 + 40) + 16LL * a3;
  v11 = *(_BYTE *)v10;
  v12 = *(_QWORD *)(v10 + 8);
  v34.m128i_i8[0] = v11;
  v34.m128i_i64[1] = v12;
  v13 = v11 ? sub_1F6C8D0(v11) : sub_1F58D40((__int64)&v34);
  if ( (v13 & 7) != 0 )
    goto LABEL_19;
  v14 = *(_WORD *)(a2 + 24);
  if ( v14 == 127 )
  {
    sub_1F738B0(
      a1,
      **(_QWORD **)(a2 + 32),
      *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL),
      (v13 >> 3) + ~a4,
      (unsigned int)(a5 + 1),
      0);
    return a1;
  }
  if ( v14 > 127 )
  {
    if ( v14 <= 144 )
    {
      if ( v14 > 141 )
      {
        v19 = *(_QWORD *)(a2 + 32);
        v20 = *(_QWORD *)v19;
        v31 = *(_QWORD *)(v19 + 8);
        v21 = sub_1F701D0(*(_QWORD *)v19, *(_DWORD *)(v19 + 8));
        if ( (v21 & 7) == 0 )
        {
          if ( v21 >> 3 > a4 )
          {
            sub_1F738B0(a1, v20, v31, a4, (unsigned int)(a5 + 1), 0);
            return a1;
          }
          if ( *(_WORD *)(a2 + 24) == 143 )
            goto LABEL_17;
        }
      }
    }
    else if ( v14 == 185 && (*(_BYTE *)(a2 + 26) & 8) == 0 && (*(_WORD *)(a2 + 26) & 0x380) == 0 )
    {
      v15 = *(_BYTE *)(a2 + 88);
      v16 = *(_QWORD *)(a2 + 96);
      v34.m128i_i8[0] = v15;
      v34.m128i_i64[1] = v16;
      v17 = v15 ? sub_1F6C8D0(v15) : sub_1F58D40((__int64)&v34);
      if ( (v17 & 7) == 0 )
      {
        if ( v17 >> 3 > a4 )
        {
          *(_BYTE *)(a1 + 16) = 1;
          *(_QWORD *)a1 = a2;
          *(_DWORD *)(a1 + 8) = a4;
          return a1;
        }
        if ( ((*(_BYTE *)(a2 + 27) ^ 0xC) & 0xC) == 0 )
        {
LABEL_17:
          *(_BYTE *)(a1 + 16) = 1;
          *(_QWORD *)a1 = 0;
          *(_DWORD *)(a1 + 8) = 0;
          return a1;
        }
      }
    }
LABEL_19:
    *(_BYTE *)(a1 + 16) = 0;
    return a1;
  }
  if ( v14 != 119 )
  {
    if ( v14 == 122 )
    {
      v22 = *(_QWORD **)(a2 + 32);
      v23 = v22[5];
      v24 = *(unsigned __int16 *)(v23 + 24);
      if ( v24 == 10 || v24 == 32 )
      {
        v25 = *(_QWORD *)(v23 + 88);
        v26 = *(_QWORD *)(v25 + 24);
        if ( *(_DWORD *)(v25 + 32) > 0x40u )
          v26 = *(_QWORD *)v26;
        if ( (v26 & 7) == 0 )
        {
          v27 = v26 >> 3;
          if ( a4 >= v27 )
          {
            sub_1F738B0(a1, *v22, v22[1], a4 - (unsigned int)v27, (unsigned int)(a5 + 1), 0);
            return a1;
          }
          goto LABEL_17;
        }
      }
    }
    goto LABEL_19;
  }
  v28 = a5 + 1;
  sub_1F738B0(&v32, **(_QWORD **)(a2 + 32), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 8LL), a4, v28, 0);
  if ( !v33 )
    goto LABEL_19;
  sub_1F738B0(&v34, *(_QWORD *)(*(_QWORD *)(a2 + 32) + 40LL), *(_QWORD *)(*(_QWORD *)(a2 + 32) + 48LL), a4, v28, 0);
  if ( !v35 )
    goto LABEL_19;
  if ( !v32.m128i_i64[0] )
  {
    v30 = _mm_loadu_si128(&v34);
    *(_BYTE *)(a1 + 16) = 1;
    *(__m128i *)a1 = v30;
    return a1;
  }
  if ( v34.m128i_i64[0] )
    goto LABEL_19;
  v29 = v33;
  *(_BYTE *)(a1 + 16) = v33;
  if ( v29 )
    *(__m128i *)a1 = _mm_loadu_si128(&v32);
  return a1;
}
