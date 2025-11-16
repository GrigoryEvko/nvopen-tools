// Function: sub_7A0070
// Address: 0x7a0070
//
__int64 __fastcall sub_7A0070(
        __int64 a1,
        __int64 a2,
        unsigned __int64 a3,
        __int64 a4,
        unsigned int a5,
        __m128i *a6,
        char *a7)
{
  char v10; // si
  unsigned int v13; // ecx
  int v14; // r9d
  __int64 v15; // rdi
  unsigned int v16; // eax
  int v17; // edx
  _QWORD *v18; // r10
  __int64 v19; // rdi
  __int64 result; // rax
  __int64 v21; // r12
  __int64 v22; // rdx
  __int64 v23; // rcx
  __int64 v24; // r8
  __int64 v25; // r9
  char v26; // al
  int v27; // ebx
  unsigned int v28; // [rsp+Ch] [rbp-44h]
  __int64 v29[7]; // [rsp+18h] [rbp-38h] BYREF

  v10 = *(_BYTE *)(a4 + 8);
  if ( (v10 & 2) != 0 )
  {
    sub_772DF0(a4, a2, a1);
    return 0;
  }
  if ( (v10 & 1) != 0 )
  {
    v21 = *(_QWORD *)(a4 + 16);
    if ( *(_BYTE *)(v21 + 173) != 6 )
      goto LABEL_18;
    v29[0] = (__int64)sub_724DC0();
    if ( sub_718E10(v21, v29[0], v22, v23, v24, v25) )
    {
      v26 = *(_BYTE *)(v29[0] + 173);
      if ( v26 == 1 )
      {
        if ( (*(_BYTE *)(v29[0] + 168) & 8) == 0 )
        {
          if ( (*(_BYTE *)(v29[0] + 171) & 4) == 0 )
          {
LABEL_54:
            *a6 = _mm_loadu_si128((const __m128i *)(v29[0] + 176));
            sub_724E30((__int64)v29);
            return 1;
          }
          goto LABEL_44;
        }
      }
      else if ( v26 == 3 )
      {
        if ( (*(_BYTE *)(v29[0] + 171) & 4) == 0 )
          goto LABEL_54;
        goto LABEL_44;
      }
      v27 = sub_79CCD0(a1, v29[0], (unsigned __int64)a6, a7, 0);
      sub_724E30((__int64)v29);
      if ( !v27 )
      {
LABEL_18:
        if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
        {
          sub_6855B0(0xA8Du, (FILE *)(a2 + 28), (_QWORD *)(a1 + 96));
          sub_770D30(a1);
          return 0;
        }
        return 0;
      }
    }
    else
    {
      if ( (unsigned __int8)(*(_BYTE *)(a3 + 140) - 9) > 2u
        || (*(_BYTE *)(a3 + 179) & 1) == 0
        || !(unsigned int)sub_8D5070(a3) )
      {
LABEL_44:
        sub_724E30((__int64)v29);
        goto LABEL_18;
      }
      sub_7790A0(a1, a6, a3, (__int64)a7);
      sub_724E30((__int64)v29);
    }
    return 1;
  }
  v13 = *(_DWORD *)(a4 + 12);
  v14 = *(_DWORD *)(a1 + 64);
  v15 = *(_QWORD *)(a1 + 56);
  v16 = v14 & v13;
  v17 = *(_DWORD *)(v15 + 4LL * (v14 & v13));
  if ( v13 )
  {
    while ( v13 != v17 )
    {
      if ( !v17 )
      {
        if ( (*(_BYTE *)(a1 + 132) & 0x20) != 0 )
          return 0;
        sub_6855B0(0xA8Cu, (FILE *)(a2 + 28), (_QWORD *)(a1 + 96));
        sub_770D30(a1);
        return 0;
      }
      v16 = v14 & (v16 + 1);
      v17 = *(_DWORD *)(v15 + 4LL * v16);
    }
  }
  v18 = *(_QWORD **)a4;
  if ( !*(_QWORD *)a4 )
  {
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xA8Au, (FILE *)(a2 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return 0;
    }
    return 0;
  }
  if ( *(char *)(a2 + 26) < 0 )
  {
    if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
    {
      sub_6855B0(0xAC0u, (FILE *)(a2 + 28), (_QWORD *)(a1 + 96));
      sub_770D30(a1);
      return 0;
    }
    return 0;
  }
  v19 = *(_QWORD *)(a4 + 24);
  if ( (*(_BYTE *)(v19 - 9) & 1) == 0
    && ((unsigned __int8)(1 << (((_BYTE)v18 - v19) & 7))
      & *(_BYTE *)(v19 + -(((unsigned int)((_DWORD)v18 - v19) >> 3) + 10))) == 0 )
  {
    if ( (unsigned __int8)(*(_BYTE *)(a3 + 140) - 9) > 2u )
    {
      if ( (*(_BYTE *)(a1 + 132) & 0x20) == 0 )
      {
        sub_6855B0(0xABFu, (FILE *)(a2 + 28), (_QWORD *)(a1 + 96));
        sub_770D30(a1);
        return 0;
      }
      return 0;
    }
    if ( (v10 & 4) == 0 )
      goto LABEL_14;
LABEL_25:
    v28 = a5;
    if ( (unsigned int)sub_773FA0(a1, a4, (FILE *)(a2 + 28)) )
    {
      v18 = *(_QWORD **)a4;
      a5 = v28;
      goto LABEL_12;
    }
    return 0;
  }
  if ( (v10 & 4) != 0 )
    goto LABEL_25;
LABEL_12:
  if ( (unsigned __int8)(*(_BYTE *)(a3 + 140) - 8) > 3u )
  {
    memcpy(a6, v18, a5);
    result = 1;
    if ( *(_BYTE *)(a3 + 140) == 6 && (a6->m128i_i8[8] & 4) != 0 )
    {
      sub_7714D0((__int64)a6);
      result = 1;
    }
    goto LABEL_15;
  }
  v19 = *(_QWORD *)(a4 + 24);
LABEL_14:
  result = sub_778F10(a1, a3, (FILE *)(a2 + 28), v18, v19, a6, (__int64)a7);
LABEL_15:
  if ( a6 == (__m128i *)a7 )
    a6[-1].m128i_i8[7] |= 1u;
  return result;
}
