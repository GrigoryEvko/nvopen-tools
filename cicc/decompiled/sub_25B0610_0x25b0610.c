// Function: sub_25B0610
// Address: 0x25b0610
//
void __fastcall sub_25B0610(_QWORD *a1, __m128i *a2, int a3, __int64 a4)
{
  __m128i *v6; // r15
  _QWORD *v7; // rbx
  __int64 v8; // r14
  __int64 v9; // rax
  __m128i v10; // xmm1
  _QWORD *v11; // rdx
  __int64 v12; // rsi
  unsigned __int64 v13; // rdi
  unsigned int v14; // eax
  _QWORD *v15; // rax
  unsigned __int64 v16; // rcx
  char v17; // r8
  unsigned int v18; // eax

  if ( !a3 )
  {
LABEL_24:
    sub_25B0490(a1, a2);
    return;
  }
  if ( a3 != 1 )
    return;
  v6 = *(__m128i **)a4;
  v7 = a1 + 1;
  v8 = *(_QWORD *)a4 + 16LL * *(unsigned int *)(a4 + 8);
  if ( *(_QWORD *)a4 == v8 )
    return;
  do
  {
    if ( (unsigned __int8)sub_25AFE60(a1, v6->m128i_i64) )
      goto LABEL_24;
    v9 = sub_22077B0(0x40u);
    v10 = _mm_loadu_si128(a2);
    v11 = (_QWORD *)a1[2];
    v12 = v9;
    *(__m128i *)(v9 + 32) = _mm_loadu_si128(v6);
    *(__m128i *)(v9 + 48) = v10;
    if ( !v11 )
    {
      v11 = v7;
      goto LABEL_26;
    }
    v13 = *(_QWORD *)(v9 + 32);
    while ( 1 )
    {
      v16 = v11[4];
      if ( v16 > v13 )
        break;
      if ( v16 == v13 )
      {
        v14 = *((_DWORD *)v11 + 10);
        if ( *(_DWORD *)(v12 + 40) < v14 || *(_DWORD *)(v12 + 40) == v14 && *(_BYTE *)(v12 + 44) < *((_BYTE *)v11 + 44) )
          break;
      }
      v15 = (_QWORD *)v11[3];
      if ( !v15 )
        goto LABEL_15;
LABEL_12:
      v11 = v15;
    }
    v15 = (_QWORD *)v11[2];
    if ( v15 )
      goto LABEL_12;
LABEL_15:
    v17 = 1;
    if ( v7 != v11 && v16 <= v13 )
    {
      v17 = 0;
      if ( v16 == v13 )
      {
        v18 = *((_DWORD *)v11 + 10);
        if ( *(_DWORD *)(v12 + 40) >= v18
          && (*(_DWORD *)(v12 + 40) != v18 || *(_BYTE *)(v12 + 44) >= *((_BYTE *)v11 + 44)) )
        {
          v17 = 0;
          goto LABEL_17;
        }
LABEL_26:
        v17 = 1;
      }
    }
LABEL_17:
    ++v6;
    sub_220F040(v17, v12, v11, v7);
    ++a1[5];
  }
  while ( (__m128i *)v8 != v6 );
}
