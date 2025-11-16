// Function: sub_CAAB30
// Address: 0xcaab30
//
__int64 __fastcall sub_CAAB30(__int64 a1)
{
  __int64 v2; // rbx
  int v3; // r13d
  _BYTE *v4; // rsi
  __int64 v5; // rax
  _BYTE *v6; // rcx
  __int64 v7; // rax
  unsigned __int64 v8; // rbx
  __m128i v9; // xmm0
  __int64 v10; // rax
  __int64 v11; // rdx
  __int64 v12; // rbx
  __int64 v13; // r8
  __int64 v14; // r9
  _QWORD *v15; // rdi
  __int64 result; // rax
  __int64 v17; // rdx
  __int64 v18; // rcx
  __int64 v19; // r8
  __m128i v20; // [rsp+8h] [rbp-68h] BYREF
  _BYTE *v21; // [rsp+18h] [rbp-58h]
  __int64 v22; // [rsp+20h] [rbp-50h]
  _QWORD v23[9]; // [rsp+28h] [rbp-48h] BYREF

  v2 = *(_QWORD *)(a1 + 40);
  v3 = *(_DWORD *)(a1 + 60);
  sub_CA7F70(a1, 1u);
  v4 = *(_BYTE **)(a1 + 40);
  v5 = *(_QWORD *)(a1 + 48);
  if ( v4 == (_BYTE *)v5 )
    goto LABEL_5;
  if ( !(unsigned __int8)sub_CA7F80(a1, v4) )
  {
    v6 = *(_BYTE **)(a1 + 40);
    if ( *v6 != 60 )
    {
      v5 = sub_CA7CD0(a1, (char *)sub_CA6130, 0, (__int64)v6);
      *(_QWORD *)(a1 + 40) = v5;
      goto LABEL_5;
    }
    sub_CA7F70(a1, 1u);
    sub_CA7D50(a1);
    result = sub_CA7E60(a1, 62, v17, v18, v19);
    if ( !(_BYTE)result )
      return result;
  }
  v5 = *(_QWORD *)(a1 + 40);
LABEL_5:
  v20.m128i_i64[0] = v2;
  v20.m128i_i64[1] = v5 - v2;
  v7 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 160) += 72LL;
  v21 = v23;
  v8 = (v7 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  v22 = 0;
  LOBYTE(v23[0]) = 0;
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
  *(_DWORD *)(v8 + 16) = 22;
  v9 = _mm_loadu_si128(&v20);
  *(_QWORD *)(v8 + 40) = v8 + 56;
  *(__m128i *)(v8 + 24) = v9;
  sub_CA64F0((__int64 *)(v8 + 40), v21, (__int64)&v21[v22]);
  v10 = *(_QWORD *)v8;
  v11 = *(_QWORD *)(a1 + 176);
  *(_QWORD *)(v8 + 8) = a1 + 176;
  v11 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v8 = v11 | v10 & 7;
  *(_QWORD *)(v11 + 8) = v8;
  v12 = *(_QWORD *)(a1 + 176) & 7LL | v8;
  *(_QWORD *)(a1 + 176) = v12;
  sub_CA80E0(a1, v12 & 0xFFFFFFFFFFFFFFF8LL, v3, 0, v13, v14);
  v15 = v21;
  *(_WORD *)(a1 + 73) = 0;
  if ( v15 != v23 )
    j_j___libc_free_0(v15, v23[0] + 1LL);
  return 1;
}
