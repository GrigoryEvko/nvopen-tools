// Function: sub_30A7B40
// Address: 0x30a7b40
//
void __fastcall sub_30A7B40(__m128i *a1, _DWORD a2, _DWORD a3, _DWORD a4, _DWORD a5, _DWORD a6, __m128i a7, char a8)
{
  __m128i v9; // xmm0
  _QWORD *v10; // r13
  _QWORD *v11; // r12
  unsigned __int64 v12; // rsi
  _QWORD *v13; // rax
  _QWORD *v14; // rdi
  __int64 v15; // rcx
  __int64 v16; // rdx
  __int64 v17; // rax
  _QWORD *v18; // rdi
  __int64 v19; // rcx
  __int64 v20; // rdx
  __int64 v21; // rax

  if ( a8 )
  {
    v9 = _mm_loadu_si128(&a7);
    a1[1].m128i_i8[0] = 1;
    *a1 = v9;
  }
  else
  {
    v10 = sub_C52410();
    v11 = v10 + 1;
    v12 = sub_C959E0();
    v13 = (_QWORD *)v10[2];
    if ( v13 )
    {
      v14 = v10 + 1;
      do
      {
        while ( 1 )
        {
          v15 = v13[2];
          v16 = v13[3];
          if ( v12 <= v13[4] )
            break;
          v13 = (_QWORD *)v13[3];
          if ( !v16 )
            goto LABEL_8;
        }
        v14 = v13;
        v13 = (_QWORD *)v13[2];
      }
      while ( v15 );
LABEL_8:
      if ( v11 != v14 && v12 >= v14[4] )
        v11 = v14;
    }
    if ( v11 == (_QWORD *)((char *)sub_C52410() + 8) )
      goto LABEL_19;
    v17 = v11[7];
    if ( !v17 )
      goto LABEL_19;
    v18 = v11 + 6;
    do
    {
      while ( 1 )
      {
        v19 = *(_QWORD *)(v17 + 16);
        v20 = *(_QWORD *)(v17 + 24);
        if ( *(_DWORD *)(v17 + 32) >= unk_502E428 )
          break;
        v17 = *(_QWORD *)(v17 + 24);
        if ( !v20 )
          goto LABEL_17;
      }
      v18 = (_QWORD *)v17;
      v17 = *(_QWORD *)(v17 + 16);
    }
    while ( v19 );
LABEL_17:
    if ( v11 + 6 == v18 || unk_502E428 < *((_DWORD *)v18 + 8) || !*((_DWORD *)v18 + 9) )
    {
LABEL_19:
      a1[1].m128i_i8[0] = 0;
    }
    else
    {
      a1->m128i_i64[0] = qword_502E468[8];
      v21 = qword_502E468[9];
      a1[1].m128i_i8[0] = 1;
      a1->m128i_i64[1] = v21;
    }
  }
}
