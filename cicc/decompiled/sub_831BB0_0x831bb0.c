// Function: sub_831BB0
// Address: 0x831bb0
//
void __fastcall sub_831BB0(__int64 a1, __int64 a2)
{
  __int64 v3; // rax
  __int64 v4; // r12
  const __m128i *v5; // r13
  const __m128i *v6; // rdi
  const __m128i *v7; // rdi
  __m128i *v8; // rax
  __m128i *v9; // rdx
  __int64 i; // rax

  if ( *(_BYTE *)(a1 + 16) != 1 )
    return;
  v3 = *(_QWORD *)(a1 + 144);
  if ( *(_BYTE *)(v3 + 24) != 3 )
    return;
  v4 = *(_QWORD *)(v3 + 56);
  if ( !v4 )
    return;
  v5 = *(const __m128i **)(v4 + 120);
  v6 = v5;
  if ( (*(_BYTE *)(v4 + 170) & 0x10) == 0 )
  {
    if ( !dword_4F077BC )
      return;
    goto LABEL_7;
  }
  if ( (unsigned int)sub_8D23E0(v5) )
    sub_8AC4A0(v4, a2);
  if ( dword_4F077BC )
  {
    v6 = *(const __m128i **)(v4 + 120);
LABEL_7:
    if ( (unsigned int)sub_8D23E0(v6) && (*(_BYTE *)(v4 + 170) & 0x10) != 0 && (*(_BYTE *)(v4 + 89) & 4) != 0 )
      sub_5EB3F0((_QWORD *)v4);
  }
  v7 = *(const __m128i **)(v4 + 120);
  if ( v5 == v7 )
    return;
  if ( *(_BYTE *)(a1 + 17) == 1 )
  {
    if ( !sub_6ED0A0(a1) )
    {
      v9 = *(__m128i **)(v4 + 120);
      *(_QWORD *)a1 = v9;
      goto LABEL_14;
    }
    v7 = *(const __m128i **)(v4 + 120);
  }
  v8 = sub_73D720(v7);
  *(_QWORD *)a1 = v8;
  v9 = v8;
LABEL_14:
  for ( i = *(_QWORD *)(a1 + 144); *(_BYTE *)(i + 24) == 1; v9 = *(__m128i **)a1 )
  {
    if ( *(_BYTE *)(i + 56) != 25 )
      break;
    *(_QWORD *)i = v9;
    i = *(_QWORD *)(i + 72);
  }
  *(_QWORD *)i = v9;
}
