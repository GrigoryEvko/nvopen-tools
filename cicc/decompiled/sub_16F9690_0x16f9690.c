// Function: sub_16F9690
// Address: 0x16f9690
//
__int64 __fastcall sub_16F9690(__int64 a1)
{
  char *v2; // rax
  __int64 v3; // r13
  unsigned __int64 v4; // r13
  char v5; // dl
  bool v6; // cc
  __int64 v7; // rax
  __m128i v8; // xmm0
  unsigned __int64 v9; // rbx
  __int64 v10; // rdx
  __int64 v11; // rdx
  __int64 v12; // rax
  _QWORD *v13; // rdi
  __m128i v15; // [rsp+8h] [rbp-58h] BYREF
  _QWORD *v16; // [rsp+18h] [rbp-48h]
  __int64 v17; // [rsp+20h] [rbp-40h]
  _QWORD v18[7]; // [rsp+28h] [rbp-38h] BYREF

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
    goto LABEL_23;
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
      goto LABEL_23;
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
LABEL_23:
      v4 = 0;
      goto LABEL_7;
    }
    v6 = v4 <= 3;
    v4 = 0;
    if ( !v6 && !v2[1] && v2[2] == -2 )
      v4 = 4LL * (v2[3] == -1);
  }
LABEL_7:
  v16 = v18;
  v17 = 0;
  LOBYTE(v18[0]) = 0;
  v15.m128i_i64[0] = (__int64)v2;
  v15.m128i_i64[1] = v4;
  v7 = sub_145CBF0((__int64 *)(a1 + 80), 72, 16);
  v8 = _mm_loadu_si128(&v15);
  v9 = v7;
  *(_QWORD *)v7 = 0;
  v10 = v17;
  *(_QWORD *)(v7 + 8) = 0;
  *(__m128i *)(v7 + 24) = v8;
  *(_DWORD *)(v7 + 16) = 1;
  *(_QWORD *)(v7 + 40) = v7 + 56;
  sub_16F6740((__int64 *)(v7 + 40), v18, (__int64)v18 + v10);
  v11 = *(_QWORD *)(a1 + 184);
  *(_QWORD *)(v9 + 8) = a1 + 184;
  v11 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v9 = v11 | *(_QWORD *)v9 & 7LL;
  *(_QWORD *)(v11 + 8) = v9;
  v12 = *(_QWORD *)(a1 + 184);
  v13 = v16;
  *(_QWORD *)(a1 + 40) += v4;
  *(_QWORD *)(a1 + 184) = v12 & 7 | v9;
  if ( v13 != v18 )
    j_j___libc_free_0(v13, v18[0] + 1LL);
  return 1;
}
