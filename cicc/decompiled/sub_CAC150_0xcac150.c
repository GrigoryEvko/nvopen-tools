// Function: sub_CAC150
// Address: 0xcac150
//
__int64 __fastcall sub_CAC150(__int64 a1)
{
  __int64 v2; // r12
  __int64 v3; // rdx
  __int64 v4; // rcx
  __int64 v5; // r8
  __int64 v6; // r13
  __int64 v7; // rax
  __int64 v8; // r15
  __int64 v9; // rax
  __int64 v11; // r13
  __int64 v12; // rax
  __int64 v13; // rax
  __int64 v14; // rax
  __int64 v15; // rax
  unsigned __int64 v16; // r12
  __int64 v17; // rax
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 v20; // rax
  _QWORD *v21; // rdi
  int v22; // [rsp+0h] [rbp-70h]
  __m128i v23; // [rsp+8h] [rbp-68h] BYREF
  _BYTE *v24; // [rsp+18h] [rbp-58h]
  __int64 v25; // [rsp+20h] [rbp-50h]
  _QWORD v26[9]; // [rsp+28h] [rbp-48h] BYREF

  sub_CA9FC0(a1, -1);
  *(_DWORD *)(a1 + 232) = 0;
  v2 = *(_QWORD *)(a1 + 40);
  *(_WORD *)(a1 + 73) = 0;
  sub_CA7E60(a1, 37, v3, v4, v5);
  v6 = *(_QWORD *)(a1 + 40);
  v7 = sub_CA7CD0(a1, (char *)sub_CA6130, 0, *(_QWORD *)(a1 + 40));
  *(_QWORD *)(a1 + 40) = v7;
  v8 = v7 - v6;
  v9 = sub_CA7CD0(a1, (char *)sub_CA6100, 0, v7);
  *(_QWORD *)(a1 + 40) = v9;
  v23 = 0u;
  v24 = v26;
  v25 = 0;
  LOBYTE(v26[0]) = 0;
  if ( v8 == 4 )
  {
    if ( *(_DWORD *)v6 != 1280131417 )
      return 0;
    v11 = a1 + 176;
    v17 = sub_CA7CD0(a1, (char *)sub_CA6130, 0, v9);
    v23.m128i_i64[0] = v2;
    *(_QWORD *)(a1 + 40) = v17;
    v23.m128i_i64[1] = v17 - v2;
    v18 = *(_QWORD *)(a1 + 80);
    *(_QWORD *)(a1 + 160) += 72LL;
    v22 = 3;
    v16 = (v18 + 15) & 0xFFFFFFFFFFFFFFF0LL;
    if ( *(_QWORD *)(a1 + 88) >= v16 + 72 && v18 )
    {
      *(_QWORD *)(a1 + 80) = v16 + 72;
      if ( !v16 )
      {
        MEMORY[8] = a1 + 176;
        BUG();
      }
      goto LABEL_14;
    }
  }
  else
  {
    if ( v8 != 3 || *(_WORD *)v6 != 16724 || *(_BYTE *)(v6 + 2) != 71 )
      return 0;
    v11 = a1 + 176;
    v12 = sub_CA7CD0(a1, (char *)sub_CA6130, 0, v9);
    *(_QWORD *)(a1 + 40) = v12;
    v13 = sub_CA7CD0(a1, (char *)sub_CA6100, 0, v12);
    *(_QWORD *)(a1 + 40) = v13;
    v14 = sub_CA7CD0(a1, (char *)sub_CA6130, 0, v13);
    v23.m128i_i64[0] = v2;
    *(_QWORD *)(a1 + 40) = v14;
    v23.m128i_i64[1] = v14 - v2;
    v15 = *(_QWORD *)(a1 + 80);
    *(_QWORD *)(a1 + 160) += 72LL;
    v22 = 4;
    v16 = (v15 + 15) & 0xFFFFFFFFFFFFFFF0LL;
    if ( *(_QWORD *)(a1 + 88) >= v16 + 72 && v15 )
    {
      *(_QWORD *)(a1 + 80) = v16 + 72;
      if ( !v16 )
      {
        MEMORY[8] = a1 + 176;
        BUG();
      }
      goto LABEL_14;
    }
  }
  v16 = sub_9D1E70(a1 + 80, 72, 72, 4);
LABEL_14:
  *(_QWORD *)v16 = 0;
  *(_QWORD *)(v16 + 8) = 0;
  *(_DWORD *)(v16 + 16) = v22;
  *(__m128i *)(v16 + 24) = _mm_loadu_si128(&v23);
  *(_QWORD *)(v16 + 40) = v16 + 56;
  sub_CA64F0((__int64 *)(v16 + 40), v24, (__int64)&v24[v25]);
  v19 = *(_QWORD *)(a1 + 176);
  v20 = *(_QWORD *)v16;
  *(_QWORD *)(v16 + 8) = v11;
  v19 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v16 = v19 | v20 & 7;
  *(_QWORD *)(v19 + 8) = v16;
  v21 = v24;
  *(_QWORD *)(a1 + 176) = *(_QWORD *)(a1 + 176) & 7LL | v16;
  if ( v21 != v26 )
    j_j___libc_free_0(v21, v26[0] + 1LL);
  return 1;
}
