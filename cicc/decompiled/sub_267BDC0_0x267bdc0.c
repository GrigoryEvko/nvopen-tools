// Function: sub_267BDC0
// Address: 0x267bdc0
//
__int64 __fastcall sub_267BDC0(__int64 a1, _QWORD **a2, __int64 a3)
{
  __int64 v5; // r14
  __int64 v6; // rax
  __int64 v7; // rdi
  __int8 *v8; // rax
  size_t v9; // rdx
  unsigned __int64 v10; // rcx
  __int64 v11; // r13
  __int64 v12; // rcx
  __int64 v13; // r8
  __int64 v14; // r9
  __m128i v15; // xmm0
  __m128i v16; // xmm1
  __m128i v17; // xmm2
  __int64 v18; // rax
  __int64 v19; // rdx
  __int64 *v20; // rdi
  __int64 v22; // rax
  __int64 v23; // rdi
  __int8 *v24; // rax
  size_t v25; // rdx
  __int64 v26; // rdx
  __int64 v27; // rcx
  __int64 v28; // r8
  __int64 v29; // r9
  __m128i v30; // xmm3
  __m128i v31; // xmm4
  __m128i v32; // xmm5
  __int64 v33; // rax
  __int64 v34[2]; // [rsp+0h] [rbp-70h] BYREF
  _QWORD v35[4]; // [rsp+10h] [rbp-60h] BYREF
  __int64 v36; // [rsp+30h] [rbp-40h] BYREF

  v5 = (*a2)[13];
  if ( *(_BYTE *)v5 == 17 )
  {
    sub_B18290(a3, "Replacing OpenMP runtime call ", 0x1Eu);
    v6 = *a2[1];
    v7 = *(_QWORD *)(v6 - 32);
    if ( v7 )
    {
      if ( *(_BYTE *)v7 )
      {
        v7 = 0;
      }
      else if ( *(_QWORD *)(v7 + 24) != *(_QWORD *)(v6 + 80) )
      {
        v7 = 0;
      }
    }
    v8 = (__int8 *)sub_BD5D20(v7);
    sub_B18290(a3, v8, v9);
    sub_B18290(a3, " with ", 6u);
    if ( *(_DWORD *)(v5 + 32) <= 0x40u )
      v10 = *(_QWORD *)(v5 + 24);
    else
      v10 = **(_QWORD **)(v5 + 24);
    sub_B16B10(v34, "FoldedValue", 11, v10);
    v11 = sub_23FD640(a3, (__int64)v34);
    sub_B18290(v11, ".", 1u);
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(v11 + 8);
    *(_BYTE *)(a1 + 12) = *(_BYTE *)(v11 + 12);
    v15 = _mm_loadu_si128((const __m128i *)(v11 + 24));
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(v11 + 16);
    *(__m128i *)(a1 + 24) = v15;
    v16 = _mm_loadu_si128((const __m128i *)(v11 + 48));
    v17 = _mm_loadu_si128((const __m128i *)(v11 + 64));
    *(_QWORD *)a1 = &unk_49D9D40;
    v18 = *(_QWORD *)(v11 + 40);
    *(__m128i *)(a1 + 48) = v16;
    *(_QWORD *)(a1 + 40) = v18;
    *(_QWORD *)(a1 + 80) = a1 + 96;
    *(_QWORD *)(a1 + 88) = 0x400000000LL;
    *(__m128i *)(a1 + 64) = v17;
    v19 = *(unsigned int *)(v11 + 88);
    if ( (_DWORD)v19 )
      sub_26781A0(a1 + 80, v11 + 80, v19, v12, v13, v14);
    v20 = (__int64 *)v35[2];
    *(_BYTE *)(a1 + 416) = *(_BYTE *)(v11 + 416);
    *(_DWORD *)(a1 + 420) = *(_DWORD *)(v11 + 420);
    *(_QWORD *)(a1 + 424) = *(_QWORD *)(v11 + 424);
    *(_QWORD *)a1 = &unk_49D9D78;
    if ( v20 != &v36 )
      j_j___libc_free_0((unsigned __int64)v20);
    if ( (_QWORD *)v34[0] != v35 )
      j_j___libc_free_0(v34[0]);
    return a1;
  }
  else
  {
    sub_B18290(a3, "Replacing OpenMP runtime call ", 0x1Eu);
    v22 = *a2[1];
    v23 = *(_QWORD *)(v22 - 32);
    if ( v23 )
    {
      if ( *(_BYTE *)v23 )
      {
        v23 = 0;
      }
      else if ( *(_QWORD *)(v23 + 24) != *(_QWORD *)(v22 + 80) )
      {
        v23 = 0;
      }
    }
    v24 = (__int8 *)sub_BD5D20(v23);
    sub_B18290(a3, v24, v25);
    sub_B18290(a3, ".", 1u);
    *(_DWORD *)(a1 + 8) = *(_DWORD *)(a3 + 8);
    *(_BYTE *)(a1 + 12) = *(_BYTE *)(a3 + 12);
    v30 = _mm_loadu_si128((const __m128i *)(a3 + 24));
    *(_QWORD *)(a1 + 16) = *(_QWORD *)(a3 + 16);
    *(__m128i *)(a1 + 24) = v30;
    v31 = _mm_loadu_si128((const __m128i *)(a3 + 48));
    v32 = _mm_loadu_si128((const __m128i *)(a3 + 64));
    *(_QWORD *)a1 = &unk_49D9D40;
    v33 = *(_QWORD *)(a3 + 40);
    *(__m128i *)(a1 + 48) = v31;
    *(_QWORD *)(a1 + 40) = v33;
    *(_QWORD *)(a1 + 80) = a1 + 96;
    *(_QWORD *)(a1 + 88) = 0x400000000LL;
    *(__m128i *)(a1 + 64) = v32;
    if ( *(_DWORD *)(a3 + 88) )
      sub_26781A0(a1 + 80, a3 + 80, v26, v27, v28, v29);
    *(_BYTE *)(a1 + 416) = *(_BYTE *)(a3 + 416);
    *(_DWORD *)(a1 + 420) = *(_DWORD *)(a3 + 420);
    *(_QWORD *)(a1 + 424) = *(_QWORD *)(a3 + 424);
    *(_QWORD *)a1 = &unk_49D9D78;
    return a1;
  }
}
