// Function: sub_CA9FC0
// Address: 0xca9fc0
//
__int64 __fastcall sub_CA9FC0(__int64 a1, int a2)
{
  int v2; // eax
  __int64 v3; // rax
  __int64 v4; // rax
  unsigned __int64 v5; // rbx
  __m128i v6; // xmm0
  __int64 v7; // rax
  __int64 v8; // rdx
  __int64 v9; // rcx
  __int64 v10; // rdx
  __m128i v12; // [rsp+18h] [rbp-68h] BYREF
  _BYTE *v13; // [rsp+28h] [rbp-58h]
  __int64 v14; // [rsp+30h] [rbp-50h]
  _QWORD v15[9]; // [rsp+38h] [rbp-48h] BYREF

  v2 = *(_DWORD *)(a1 + 68);
  v13 = v15;
  v14 = 0;
  LOBYTE(v15[0]) = 0;
  if ( !v2 && a2 < *(_DWORD *)(a1 + 56) )
  {
    do
    {
      v3 = *(_QWORD *)(a1 + 40);
      *(_QWORD *)(a1 + 160) += 72LL;
      v12.m128i_i64[0] = v3;
      v4 = *(_QWORD *)(a1 + 80);
      v12.m128i_i64[1] = 1;
      v5 = (v4 + 15) & 0xFFFFFFFFFFFFFFF0LL;
      if ( *(_QWORD *)(a1 + 88) >= v5 + 72 && v4 )
      {
        *(_QWORD *)(a1 + 80) = v5 + 72;
        if ( !v5 )
        {
          MEMORY[8] = a1 + 176;
          BUG();
        }
      }
      else
      {
        v5 = sub_9D1E70(a1 + 80, 72, 72, 4);
      }
      *(_QWORD *)v5 = 0;
      *(_QWORD *)(v5 + 8) = 0;
      *(_DWORD *)(v5 + 16) = 8;
      v6 = _mm_loadu_si128(&v12);
      *(_QWORD *)(v5 + 40) = v5 + 56;
      *(__m128i *)(v5 + 24) = v6;
      sub_CA64F0((__int64 *)(v5 + 40), v13, (__int64)&v13[v14]);
      v7 = *(_QWORD *)v5;
      v8 = *(_QWORD *)(a1 + 176);
      *(_QWORD *)(v5 + 8) = a1 + 176;
      v8 &= 0xFFFFFFFFFFFFFFF8LL;
      *(_QWORD *)v5 = v8 | v7 & 7;
      *(_QWORD *)(v8 + 8) = v5;
      v9 = *(unsigned int *)(a1 + 200);
      v10 = *(_QWORD *)(a1 + 192);
      LODWORD(v7) = *(_DWORD *)(a1 + 200);
      *(_QWORD *)(a1 + 176) = *(_QWORD *)(a1 + 176) & 7LL | v5;
      LODWORD(v10) = *(_DWORD *)(v10 + 4 * v9 - 4);
      *(_DWORD *)(a1 + 200) = v7 - 1;
      *(_DWORD *)(a1 + 56) = v10;
    }
    while ( a2 < (int)v10 );
    if ( v13 != (_BYTE *)v15 )
      j_j___libc_free_0(v13, v15[0] + 1LL);
  }
  return 1;
}
