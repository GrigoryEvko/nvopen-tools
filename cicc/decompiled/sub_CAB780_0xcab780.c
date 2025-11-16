// Function: sub_CAB780
// Address: 0xcab780
//
__int64 __fastcall sub_CAB780(__int64 a1, __int64 a2, __int64 a3, __int64 a4, __int64 a5, __int64 a6)
{
  unsigned __int64 *v6; // r15
  unsigned int v8; // edx
  unsigned __int64 *v9; // r12
  __int64 v10; // rcx
  __int64 v11; // rax
  unsigned int v12; // r8d
  int v14; // eax
  __int64 v15; // r14
  __int64 v16; // rax
  __int64 v17; // rax
  unsigned __int64 v18; // r12
  __m128i v19; // xmm0
  __int64 v20; // rdx
  __int64 v21; // rax
  _QWORD *v22; // rdi
  __int64 v23; // rax
  unsigned __int64 v24; // rcx
  __m128i v25; // xmm2
  unsigned __int64 v26; // rax
  unsigned __int64 v27; // rdx
  __int64 v28; // r9
  _QWORD *v29; // rdi
  __int64 v30; // rax
  unsigned int v31; // [rsp+4h] [rbp-7Ch]
  unsigned __int64 v32; // [rsp+8h] [rbp-78h]
  unsigned int v33; // [rsp+8h] [rbp-78h]
  __m128i v34; // [rsp+18h] [rbp-68h] BYREF
  _BYTE *v35; // [rsp+28h] [rbp-58h]
  __int64 v36; // [rsp+30h] [rbp-50h]
  _QWORD v37[9]; // [rsp+38h] [rbp-48h] BYREF

  v6 = (unsigned __int64 *)(a1 + 176);
  v8 = *(_DWORD *)(a1 + 232);
  if ( !v8 )
  {
    v14 = *(_DWORD *)(a1 + 68);
    if ( !v14 )
    {
      sub_CAB2C0(a1, *(_DWORD *)(a1 + 60), 10, (unsigned __int64 *)(a1 + 176), a5, a6);
      v14 = *(_DWORD *)(a1 + 68);
    }
    v15 = a1 + 80;
    *(_BYTE *)(a1 + 73) = v14 == 0;
    goto LABEL_11;
  }
  v9 = *(unsigned __int64 **)(a1 + 184);
  v10 = *(_QWORD *)(a1 + 224) + 24LL * v8 - 24;
  v11 = *(_QWORD *)v10;
  v12 = *(_DWORD *)(v10 + 8);
  *(_DWORD *)(a1 + 232) = v8 - 1;
  v35 = v37;
  v36 = 0;
  LOBYTE(v37[0]) = 0;
  v34 = _mm_loadu_si128((const __m128i *)(v11 + 24));
  if ( v9 == v6 )
  {
LABEL_7:
    *(_BYTE *)(a1 + 75) = 1;
    return 0;
  }
  while ( (unsigned __int64 *)v11 != v9 )
  {
    v9 = (unsigned __int64 *)v9[1];
    if ( v9 == v6 )
      goto LABEL_7;
  }
  v23 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 160) += 72LL;
  v15 = a1 + 80;
  v24 = (v23 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( *(_QWORD *)(a1 + 88) >= v24 + 72 && v23 )
  {
    *(_QWORD *)(a1 + 80) = v24 + 72;
    if ( !v24 )
    {
      MEMORY[8] = v9;
      BUG();
    }
  }
  else
  {
    v33 = v12;
    v30 = sub_9D1E70(a1 + 80, 72, 72, 4);
    v12 = v33;
    v24 = v30;
  }
  *(_QWORD *)v24 = 0;
  *(_QWORD *)(v24 + 8) = 0;
  v31 = v12;
  *(_DWORD *)(v24 + 16) = 16;
  v25 = _mm_loadu_si128(&v34);
  *(_QWORD *)(v24 + 40) = v24 + 56;
  *(__m128i *)(v24 + 24) = v25;
  v32 = v24;
  sub_CA64F0((__int64 *)(v24 + 40), v35, (__int64)&v35[v36]);
  v26 = *(_QWORD *)v32;
  v27 = *v9 & 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)(v32 + 8) = v9;
  *(_QWORD *)v32 = v27 | v26 & 7;
  *(_QWORD *)(v27 + 8) = v32;
  *v9 = v32 | *v9 & 7;
  sub_CAB2C0(a1, v31, 10, (unsigned __int64 *)v32, v31, v28);
  v29 = v35;
  *(_BYTE *)(a1 + 73) = 0;
  if ( v29 != v37 )
    j_j___libc_free_0(v29, v37[0] + 1LL);
LABEL_11:
  *(_BYTE *)(a1 + 74) = 0;
  v16 = *(_QWORD *)(a1 + 40);
  v35 = v37;
  v34.m128i_i64[0] = v16;
  v36 = 0;
  LOBYTE(v37[0]) = 0;
  v34.m128i_i64[1] = 1;
  sub_CA7F70(a1, 1u);
  v17 = *(_QWORD *)(a1 + 80);
  *(_QWORD *)(a1 + 160) += 72LL;
  v18 = (v17 + 15) & 0xFFFFFFFFFFFFFFF0LL;
  if ( *(_QWORD *)(a1 + 88) >= v18 + 72 && v17 )
  {
    *(_QWORD *)(a1 + 80) = v18 + 72;
    if ( !v18 )
    {
      MEMORY[8] = v6;
      BUG();
    }
  }
  else
  {
    v18 = sub_9D1E70(v15, 72, 72, 4);
  }
  *(_QWORD *)v18 = 0;
  *(_QWORD *)(v18 + 8) = 0;
  *(_DWORD *)(v18 + 16) = 17;
  v19 = _mm_loadu_si128(&v34);
  *(_QWORD *)(v18 + 40) = v18 + 56;
  *(__m128i *)(v18 + 24) = v19;
  sub_CA64F0((__int64 *)(v18 + 40), v35, (__int64)&v35[v36]);
  v20 = *(_QWORD *)(a1 + 176);
  v21 = *(_QWORD *)v18;
  *(_QWORD *)(v18 + 8) = v6;
  v20 &= 0xFFFFFFFFFFFFFFF8LL;
  *(_QWORD *)v18 = v20 | v21 & 7;
  *(_QWORD *)(v20 + 8) = v18;
  v22 = v35;
  *(_QWORD *)(a1 + 176) = *(_QWORD *)(a1 + 176) & 7LL | v18;
  if ( v22 != v37 )
    j_j___libc_free_0(v22, v37[0] + 1LL);
  return 1;
}
