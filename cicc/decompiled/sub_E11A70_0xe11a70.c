// Function: sub_E11A70
// Address: 0xe11a70
//
__int64 __fastcall sub_E11A70(__int64 a1, char **a2)
{
  char *v4; // rsi
  unsigned __int64 v5; // rax
  char *v6; // rdi
  unsigned __int64 v7; // rsi
  unsigned __int64 v8; // rax
  __int64 v9; // rax
  __m128i si128; // xmm0
  __m128i *v11; // rdi
  _BYTE *v12; // r14
  char *v13; // rsi
  unsigned __int64 v14; // rax
  __int64 v15; // rdi
  unsigned __int64 v16; // rsi
  unsigned __int64 v17; // rax
  __int64 v18; // rax
  _BYTE *v19; // r13
  __int64 result; // rax

  v4 = a2[1];
  v5 = (unsigned __int64)a2[2];
  v6 = *a2;
  if ( (unsigned __int64)(v4 + 24) > v5 )
  {
    v7 = (unsigned __int64)(v4 + 1016);
    v8 = 2 * v5;
    if ( v7 <= v8 )
      a2[2] = (char *)v8;
    else
      a2[2] = (char *)v7;
    v9 = realloc(v6);
    *a2 = (char *)v9;
    v6 = (char *)v9;
    if ( !v9 )
      goto LABEL_18;
    v4 = a2[1];
  }
  si128 = _mm_load_si128((const __m128i *)&xmmword_3F7C280);
  v11 = (__m128i *)&v6[(_QWORD)v4];
  v11[1].m128i_i64[0] = 0x20726F6620656C62LL;
  *v11 = si128;
  a2[1] += 24;
  v12 = *(_BYTE **)(a1 + 16);
  (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v12 + 32LL))(v12, a2);
  if ( (v12[9] & 0xC0) != 0x40 )
    (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v12 + 40LL))(v12, a2);
  v13 = a2[1];
  v14 = (unsigned __int64)a2[2];
  v15 = (__int64)*a2;
  if ( (unsigned __int64)(v13 + 4) > v14 )
  {
    v16 = (unsigned __int64)(v13 + 996);
    v17 = 2 * v14;
    if ( v16 > v17 )
      a2[2] = (char *)v16;
    else
      a2[2] = (char *)v17;
    v18 = realloc((void *)v15);
    *a2 = (char *)v18;
    v15 = v18;
    if ( v18 )
    {
      v13 = a2[1];
      goto LABEL_13;
    }
LABEL_18:
    abort();
  }
LABEL_13:
  *(_DWORD *)&v13[v15] = 762210605;
  a2[1] += 4;
  v19 = *(_BYTE **)(a1 + 24);
  (*(void (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v19 + 32LL))(v19, a2);
  result = v19[9] & 0xC0;
  if ( (v19[9] & 0xC0) != 0x40 )
    return (*(__int64 (__fastcall **)(_BYTE *, char **))(*(_QWORD *)v19 + 40LL))(v19, a2);
  return result;
}
