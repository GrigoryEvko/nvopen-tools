// Function: sub_7781C0
// Address: 0x7781c0
//
__int64 __fastcall sub_7781C0(
        __int64 a1,
        __int64 a2,
        __int64 a3,
        const __m128i *a4,
        __m128i *a5,
        __m128i *a6,
        __int64 a7)
{
  unsigned __int64 v10; // rax
  char v11; // cl
  unsigned __int64 v12; // r8
  unsigned __int64 v13; // r12
  __int64 v14; // rax
  unsigned __int64 v15; // r12
  __int64 v16; // rax
  unsigned __int8 v17; // al
  __int64 i; // rax
  __int64 v19; // rax
  __int64 result; // rax
  unsigned int v21; // eax
  __int64 v23; // [rsp+20h] [rbp-70h]
  unsigned __int64 v24; // [rsp+28h] [rbp-68h]
  unsigned __int64 v25; // [rsp+30h] [rbp-60h]
  int v26; // [rsp+38h] [rbp-58h]
  unsigned int v27; // [rsp+3Ch] [rbp-54h]
  char v28; // [rsp+40h] [rbp-50h]
  unsigned __int64 v29; // [rsp+40h] [rbp-50h]
  unsigned int v30; // [rsp+54h] [rbp-3Ch] BYREF
  int v31; // [rsp+58h] [rbp-38h] BYREF
  int v32[13]; // [rsp+5Ch] [rbp-34h] BYREF

  v10 = *(_QWORD *)(a3 + 160);
  v11 = *(_BYTE *)(v10 + 140);
  v30 = 1;
  v25 = v10;
  if ( v11 == 12 )
  {
    do
    {
      v10 = *(_QWORD *)(v10 + 160);
      v11 = *(_BYTE *)(v10 + 140);
    }
    while ( v11 == 12 );
    v25 = v10;
  }
  v27 = 16;
  v12 = *(_QWORD *)(a3 + 128);
  v13 = *(_QWORD *)(v25 + 128);
  v24 = v12 / v13;
  if ( (unsigned __int8)(v11 - 2) > 1u )
  {
    v29 = *(_QWORD *)(a3 + 128);
    v21 = sub_7764B0(a1, v25, &v30);
    v12 = v29;
    v27 = v21;
  }
  v26 = 0;
  v28 = *(_BYTE *)(v25 + 140);
  v14 = 13;
  if ( v28 == 2 )
  {
    v14 = *(unsigned __int8 *)(v25 + 160);
    v26 = byte_4B6DF90[v14];
  }
  if ( v12 < v13 )
    return v30;
  v15 = 0;
  v23 = 16 * v14 + 82863808;
  while ( 1 )
  {
    v17 = *(_BYTE *)(a2 + 56);
    if ( v28 == 2 )
    {
      v32[0] = 0;
      switch ( v17 )
      {
        case '\'':
          *a6 = _mm_loadu_si128(a4);
          sub_621270((unsigned __int16 *)a6, a5->m128i_i16, v26, (_BOOL4 *)v32);
          goto LABEL_29;
        case '(':
          *a6 = _mm_loadu_si128(a4);
          sub_6215F0((unsigned __int16 *)a6, a5->m128i_i16, v26, (_BOOL4 *)v32);
          goto LABEL_29;
        case ')':
          *a6 = _mm_loadu_si128(a4);
          sub_621F20(a6, a5, v26, (_BOOL4 *)v32);
          goto LABEL_29;
        case '*':
          *a6 = _mm_loadu_si128(a4);
          sub_6220A0(a6, a5, v26, (_BOOL4 *)v32);
          goto LABEL_29;
        case '+':
          *a6 = _mm_loadu_si128(a4);
          sub_6220C0(a6, a5, v26, (_BOOL4 *)v32);
          goto LABEL_29;
        case '5':
        case '6':
          for ( i = **(_QWORD **)(*(_QWORD *)(a2 + 72) + 16LL); *(_BYTE *)(i + 140) == 12; i = *(_QWORD *)(i + 160) )
            ;
          v19 = sub_620EE0(a5, byte_4B6DF90[*(unsigned __int8 *)(i + 160)], v32);
          if ( v32[0] )
          {
            v30 = 0;
            if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
            {
              sub_6855B0(0x3Du, (FILE *)(a2 + 28), (_QWORD *)(a1 + 96));
              sub_770D30(a1);
            }
          }
          else if ( v19 < 0 )
          {
            v30 = 0;
            if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
            {
              sub_6855B0(0xA91u, (FILE *)(a2 + 28), (_QWORD *)(a1 + 96));
              sub_770D30(a1);
            }
          }
          else if ( *(_QWORD *)(a3 + 128) * dword_4F06BA0 <= v19 )
          {
            v30 = 0;
            if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
            {
              sub_67E440(0xA92u, (_DWORD *)(a2 + 28), v19, (_QWORD *)(a1 + 96));
              sub_770D30(a1);
            }
          }
          else
          {
            *a6 = _mm_loadu_si128(a4);
            if ( *(_BYTE *)(a2 + 56) == 53 )
            {
              if ( v26 && a6->m128i_i16[0] < 0 )
              {
                if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
                {
                  sub_6855B0(0xB54u, (FILE *)(a2 + 28), (_QWORD *)(a1 + 96));
                  sub_770D30(a1);
                }
                v30 = 0;
              }
              else
              {
                sub_621410((__int64)a6, v19, v32);
                if ( !v26 )
                  goto LABEL_27;
              }
              goto LABEL_30;
            }
            sub_6214E0(a6->m128i_i16, v19, v26, dword_4F0699C);
          }
LABEL_29:
          if ( !v26 )
          {
LABEL_27:
            sub_6213D0((__int64)a6, v23);
            goto LABEL_15;
          }
LABEL_30:
          sub_6215A0(a6->m128i_i16, *(_DWORD *)(v25 + 128) * dword_4F06BA0);
          break;
        case '7':
          *a6 = _mm_loadu_si128(a4);
          sub_6213D0((__int64)a6, (__int64)a5);
          goto LABEL_29;
        case '8':
          *a6 = _mm_loadu_si128(a4);
          sub_6213B0((__int64)a6, (__int64)a5);
          goto LABEL_29;
        case '9':
          *a6 = _mm_loadu_si128(a4);
          sub_6213F0((__int64)a6, (__int64)a5);
          goto LABEL_29;
        default:
          goto LABEL_54;
      }
      goto LABEL_15;
    }
    v31 = 0;
    if ( v17 == 41 )
    {
      sub_70BBE0(*(_BYTE *)(v25 + 160), a4, a4, a6, &v31, v32);
LABEL_14:
      if ( v31 )
        break;
      goto LABEL_15;
    }
    if ( v17 > 0x29u )
    {
      if ( v17 != 42 )
LABEL_54:
        sub_721090();
      sub_70BCF0(*(_BYTE *)(v25 + 160), a4, a4, a6, &v31, v32);
      goto LABEL_14;
    }
    if ( v17 == 39 )
    {
      sub_70B8D0(*(_BYTE *)(v25 + 160), a4, a4, a6, &v31, v32);
      goto LABEL_14;
    }
    if ( v17 != 40 )
      goto LABEL_54;
    sub_70B9E0(*(_BYTE *)(v25 + 160), a4, a4, a6, &v31, v32);
    if ( v31 )
      break;
LABEL_15:
    ++v15;
    v16 = -(((unsigned int)((_DWORD)a6 - a7) >> 3) + 10);
    *(_BYTE *)(a7 + v16) |= 1 << (((_BYTE)a6 - a7) & 7);
    a4 = (const __m128i *)((char *)a4 + v27);
    a5 = (__m128i *)((char *)a5 + v27);
    a6 = (__m128i *)((char *)a6 + v27);
    if ( v24 <= v15 )
      return v30;
  }
  v30 = 0;
  result = 0;
  if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
  {
    sub_6855B0(0xA94u, (FILE *)(a2 + 28), (_QWORD *)(a1 + 96));
    sub_770D30(a1);
    return v30;
  }
  return result;
}
