// Function: sub_7103C0
// Address: 0x7103c0
//
unsigned int *__fastcall sub_7103C0(__int64 a1, __int64 a2, _DWORD *a3, unsigned __int8 *a4, _DWORD *a5, int a6)
{
  __int64 v8; // rax
  char i; // dl
  unsigned __int8 v10; // r15
  const __m128i *v11; // r12
  int v12; // eax
  _BOOL4 v13; // eax
  int v14; // edx
  int v15; // ecx
  unsigned int *result; // rax
  char v17; // dl
  const __m128i *v18; // rax
  __m128i v19; // xmm1
  _DWORD *v21; // [rsp+18h] [rbp-88h]
  _DWORD *v22; // [rsp+18h] [rbp-88h]
  _BOOL4 v23; // [rsp+2Ch] [rbp-74h] BYREF
  __m128i v24; // [rsp+30h] [rbp-70h] BYREF
  __int128 v25; // [rsp+40h] [rbp-60h] BYREF
  __m128i v26; // [rsp+50h] [rbp-50h] BYREF
  __m128i v27; // [rsp+60h] [rbp-40h]

  v8 = *(_QWORD *)(a1 + 128);
  for ( i = *(_BYTE *)(v8 + 140); i == 12; i = *(_BYTE *)(v8 + 140) )
    v8 = *(_QWORD *)(v8 + 160);
  v10 = *(_BYTE *)(v8 + 160);
  if ( i == 5 )
  {
    v18 = *(const __m128i **)(a1 + 176);
    v11 = &v26;
    if ( *(_BYTE *)(a1 + 173) == 4 )
    {
      v26 = _mm_loadu_si128(v18);
      v27 = _mm_loadu_si128(v18 + 1);
    }
    else
    {
      v19 = _mm_loadu_si128((const __m128i *)(v18[7].m128i_i64[1] + 176));
      v26 = _mm_loadu_si128(v18 + 11);
      v27 = v19;
    }
  }
  else
  {
    v11 = (const __m128i *)(a1 + 176);
    if ( i == 4 )
    {
      v11 = (const __m128i *)&v25;
      v22 = a5;
      sub_70B680(*(_BYTE *)(v8 + 160), 0, &v25, &v23);
      a5 = v22;
    }
  }
  *a3 = 0;
  *a4 = 5;
  v21 = a5;
  v12 = sub_620E90(a2);
  v13 = sub_710280(v11, v10, &v24, v12, v21);
  v23 = !v13;
  if ( v13 )
  {
    v14 = 0;
  }
  else
  {
    v14 = 1;
    v15 = dword_4F077C0;
    if ( !dword_4F077C0 )
      goto LABEL_8;
    if ( qword_4F077A8 <= 0x76BFu )
    {
LABEL_17:
      v15 = 0;
      goto LABEL_8;
    }
    sub_70B790(v10, v11, &v24, a2);
    v14 = v23;
  }
  if ( !dword_4F077C0 )
    goto LABEL_17;
  v15 = 1;
  if ( qword_4F077A8 <= 0x76BFu )
    goto LABEL_17;
LABEL_8:
  sub_70FF50(&v24, a2, v14 == 0, v15, a3, a4);
  result = &dword_4F077C0;
  if ( v23 || *a3 )
  {
    *a3 = 173;
    v17 = 8;
    if ( a6 )
      v17 = dword_4F077C0 == 0 ? 8 : 5;
    *a4 = v17;
  }
  return result;
}
