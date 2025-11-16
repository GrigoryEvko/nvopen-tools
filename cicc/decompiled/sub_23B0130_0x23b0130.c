// Function: sub_23B0130
// Address: 0x23b0130
//
__int64 __fastcall sub_23B0130(int *a1, __int64 a2)
{
  __int64 v4; // rax
  _BYTE *v5; // rsi
  __int64 v6; // r13
  __int64 v7; // rdx
  __m128i v8; // xmm1
  int v9; // eax
  __int64 v10; // rdi
  int *v11; // r12
  __int64 v12; // rbx
  __int64 v13; // r14
  int v14; // eax
  __int64 v15; // rdi

  v4 = sub_22077B0(0x58u);
  v5 = (_BYTE *)*((_QWORD *)a1 + 5);
  v6 = v4;
  v7 = (__int64)&v5[*((_QWORD *)a1 + 6)];
  *(_DWORD *)(v4 + 32) = a1[8];
  *(_QWORD *)(v4 + 40) = v4 + 56;
  sub_23AEDD0((__int64 *)(v4 + 40), v5, v7);
  v8 = _mm_loadu_si128((const __m128i *)(a1 + 18));
  v9 = *a1;
  *(_QWORD *)(v6 + 8) = a2;
  v10 = *((_QWORD *)a1 + 3);
  *(_QWORD *)(v6 + 16) = 0;
  *(_DWORD *)v6 = v9;
  *(_QWORD *)(v6 + 24) = 0;
  *(__m128i *)(v6 + 72) = v8;
  if ( v10 )
    *(_QWORD *)(v6 + 24) = sub_23B0130(v10, v6);
  v11 = (int *)*((_QWORD *)a1 + 2);
  if ( v11 )
  {
    v12 = v6;
    do
    {
      v13 = v12;
      v12 = sub_22077B0(0x58u);
      *(_DWORD *)(v12 + 32) = v11[8];
      *(_QWORD *)(v12 + 40) = v12 + 56;
      sub_23AEDD0((__int64 *)(v12 + 40), *((_BYTE **)v11 + 5), *((_QWORD *)v11 + 5) + *((_QWORD *)v11 + 6));
      *(__m128i *)(v12 + 72) = _mm_loadu_si128((const __m128i *)(v11 + 18));
      v14 = *v11;
      *(_QWORD *)(v12 + 16) = 0;
      *(_DWORD *)v12 = v14;
      *(_QWORD *)(v12 + 24) = 0;
      *(_QWORD *)(v13 + 16) = v12;
      *(_QWORD *)(v12 + 8) = v13;
      v15 = *((_QWORD *)v11 + 3);
      if ( v15 )
        *(_QWORD *)(v12 + 24) = sub_23B0130(v15, v12);
      v11 = (int *)*((_QWORD *)v11 + 2);
    }
    while ( v11 );
  }
  return v6;
}
