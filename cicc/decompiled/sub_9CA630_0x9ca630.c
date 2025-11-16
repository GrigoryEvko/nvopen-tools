// Function: sub_9CA630
// Address: 0x9ca630
//
__int64 __fastcall sub_9CA630(_QWORD *a1, __int64 *a2)
{
  _QWORD *v2; // r15
  _QWORD *v4; // r12
  unsigned __int64 v5; // rcx
  unsigned __int64 v6; // rdx
  _QWORD *v7; // rax
  _BOOL4 v8; // r8d
  __int64 v9; // rax
  __m128i v10; // xmm1
  __int64 v11; // r13
  unsigned __int64 v12; // rax
  __m128i v13; // xmm2
  __int64 v14; // rdx
  __int64 v15; // rax
  __int64 v16; // rax
  int v17; // esi
  __int64 v18; // rax
  _BOOL4 v20; // [rsp+Ch] [rbp-34h]

  v2 = a1 + 1;
  v4 = (_QWORD *)a1[2];
  if ( v4 )
  {
    v5 = *a2;
    while ( 1 )
    {
      v6 = v4[4];
      v7 = (_QWORD *)v4[3];
      if ( v5 < v6 )
        v7 = (_QWORD *)v4[2];
      if ( !v7 )
        break;
      v4 = v7;
    }
    v8 = 1;
    if ( v2 != v4 )
      v8 = v5 < v6;
  }
  else
  {
    v4 = a1 + 1;
    v8 = 1;
  }
  v20 = v8;
  v9 = sub_22077B0(144);
  v10 = _mm_loadu_si128((const __m128i *)(a2 + 3));
  v11 = v9;
  v12 = *a2;
  v13 = _mm_loadu_si128((const __m128i *)(a2 + 5));
  *(__m128i *)(v11 + 40) = _mm_loadu_si128((const __m128i *)(a2 + 1));
  v14 = v11 + 104;
  *(_QWORD *)(v11 + 32) = v12;
  v15 = a2[7];
  *(__m128i *)(v11 + 56) = v10;
  *(_QWORD *)(v11 + 88) = v15;
  v16 = a2[10];
  *(__m128i *)(v11 + 72) = v13;
  if ( v16 )
  {
    v17 = *((_DWORD *)a2 + 18);
    *(_QWORD *)(v11 + 112) = v16;
    *(_DWORD *)(v11 + 104) = v17;
    *(_QWORD *)(v11 + 120) = a2[11];
    *(_QWORD *)(v11 + 128) = a2[12];
    *(_QWORD *)(v16 + 8) = v14;
    v18 = a2[13];
    a2[10] = 0;
    *(_QWORD *)(v11 + 136) = v18;
    a2[11] = (__int64)(a2 + 9);
    a2[12] = (__int64)(a2 + 9);
    a2[13] = 0;
  }
  else
  {
    *(_DWORD *)(v11 + 104) = 0;
    *(_QWORD *)(v11 + 112) = 0;
    *(_QWORD *)(v11 + 120) = v14;
    *(_QWORD *)(v11 + 128) = v14;
    *(_QWORD *)(v11 + 136) = 0;
  }
  sub_220F040(v20, v11, v4, v2);
  ++a1[5];
  return v11;
}
