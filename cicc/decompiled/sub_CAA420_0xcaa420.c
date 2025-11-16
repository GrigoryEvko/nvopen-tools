// Function: sub_CAA420
// Address: 0xcaa420
//
__int64 __fastcall sub_CAA420(__int64 a1)
{
  char *v2; // rax
  __int64 v3; // r13
  unsigned __int64 v4; // r13
  char v5; // dl
  bool v6; // cc
  __int64 v7; // rax
  unsigned __int64 v8; // rbx
  __m128i v9; // xmm0
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rax
  _QWORD *v13; // rdi
  __m128i v15; // [rsp+8h] [rbp-68h] BYREF
  _BYTE *v16; // [rsp+18h] [rbp-58h]
  __int64 v17; // [rsp+20h] [rbp-50h]
  _QWORD v18[9]; // [rsp+28h] [rbp-48h] BYREF

  v2 = *(char **)(a1 + 40);
  v3 = *(_QWORD *)(a1 + 48);
  *(_BYTE *)(a1 + 72) = 0;
  v4 = v3 - (_QWORD)v2;
  if ( !v4 )
    goto LABEL_7;
  v5 = *v2;
  if ( *v2 == -2 )
  {
    if ( v4 != 1 )
    {
      v4 = 2LL * (v2[1] == -1);
      goto LABEL_7;
    }
    goto LABEL_27;
  }
  if ( v5 == -1 )
  {
    if ( v4 <= 3 )
    {
      if ( v4 != 1 )
      {
        v4 = 2LL * (v2[1] == -2);
        goto LABEL_7;
      }
      goto LABEL_27;
    }
    v4 = 0;
    if ( v2[1] == -2 )
    {
      v4 = 2;
      if ( !v2[2] )
        v4 = v2[3] == 0 ? 4LL : 2LL;
    }
  }
  else
  {
    if ( v5 )
    {
      if ( v5 == -17 )
      {
        v6 = v4 <= 2;
        v4 = 0;
        if ( !v6 && v2[1] == -69 )
          v4 = 3LL * (v2[2] == -65);
        goto LABEL_7;
      }
LABEL_27:
      v4 = 0;
      goto LABEL_7;
    }
    v6 = v4 <= 3;
    v4 = 0;
    if ( !v6 && !v2[1] && v2[2] == -2 )
      v4 = 4LL * (v2[3] == -1);
  }
LABEL_7:
  v15.m128i_i64[0] = (__int64)v2;
  v7 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 160) += 72LL;
  v16 = v18;
  v8 = (v7 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  v17 = 0;
  LOBYTE(v18[0]) = 0;
  v15.m128i_i64[1] = v4;
  if ( *(_QWORD *)(a1 + 88) >= v8 + 72 && v7 )
  {
    *(_QWORD *)(a1 + 80) = v8 + 72;
    if ( !v8 )
    {
      MEMORY[8] = a1 + 176;
      BUG();
    }
  }
  else
  {
    v8 = sub_9D1E70(a1 + 80, 72, 72, 4);
  }
  *(_QWORD *)v8 = 0;
  *(_QWORD *)(v8 + 8) = 0;
  *(_DWORD *)(v8 + 16) = 1;
  v9 = _mm_loadu_si128(&v15);
  *(_QWORD *)(v8 + 40) = v8 + 56;
  *(__m128i *)(v8 + 24) = v9;
  sub_CA64F0((__int64 *)(v8 + 40), v16, (__int64)&v16[v17]);
  v10 = *(_QWORD *)v8;
  v11 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(v8 + 8) = a1 + 176;
  v11 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v8 = v11 | v10 & 7;
  *(_QWORD *)(v11 + 8) = v8;
  v12 = *(_QWORD *)(a1 + 176);
  v13 = v16;
  *(_QWORD *)(a1 + 40) += v4;
  *(_QWORD *)(a1 + 176) = v12 & 7 | v8;
  if ( v13 != v18 )
    j_j___libc_free_0(v13, v18[0] + 1LL);
  return 1;
}
